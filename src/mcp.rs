//! Model Context Protocol client for the chat page: starts the servers of an `mcpServers` config file (stdio
//! commands or streamable-HTTP addresses), lists their tools and calls them. Only tools; no resources, prompts,
//! sampling or OAuth.
use anyhow::{Context, anyhow, bail};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

const VERSION: &str = "2025-06-18";

#[derive(Clone, Debug, PartialEq)]
pub enum Spec {
    Stdio { command: String, args: Vec<String>, env: Vec<(String, String)> },
    Http { url: String, headers: Vec<(String, String)> },
}

/// The servers of a config file in the format other MCP clients use:
/// `{"mcpServers": {"name": {"command", "args", "env"} | {"url", "headers"}}}`, in key order.
pub fn parse(text: &str) -> anyhow::Result<Vec<(String, Spec)>> {
    // serde_json is built without `preserve_order`, so the servers come out in key order.
    let v: Value = serde_json::from_str(text).context("MCP config is not JSON")?;
    let strs = |v: &Value| -> Vec<(String, String)> {
        v.as_object().into_iter().flatten().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect()
    };
    let mut out = Vec::new();
    for (name, s) in v["mcpServers"].as_object().into_iter().flatten() {
        out.push((name.clone(), match (s["command"].as_str(), s["url"].as_str()) {
            (Some(c), _) => Spec::Stdio { command: c.into(), env: strs(&s["env"]),
                args: s["args"].as_array().into_iter().flatten().filter_map(|a| a.as_str().map(String::from)).collect() },
            (None, Some(u)) if u.starts_with("http://") || u.starts_with("https://") => Spec::Http { url: u.into(), headers: strs(&s["headers"]) },
            (None, Some(u)) => bail!("MCP server {name:?}: url {u:?} is not http or https"),
            _ => bail!("MCP server {name:?} has neither \"command\" nor \"url\""),
        }));
    }
    Ok(out)
}

/// Seconds to wait for one answer [default 120; `STRATA_MCP_TIMEOUT`]. The first start of an `npx`/`uvx` server
/// downloads its package, so the default is long.
fn timeout() -> Duration {
    Duration::from_secs(std::env::var("STRATA_MCP_TIMEOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(120))
}

enum Conn {
    /// `rx` carries the child's stdout lines from a reader thread, so a read can time out.
    Stdio { child: std::process::Child, stdin: std::process::ChildStdin, rx: Receiver<String> },
    Http { agent: ureq::Agent, url: String, headers: Vec<(String, String)>, session: Option<String> },
}

impl Drop for Conn {
    fn drop(&mut self) {
        if let Conn::Stdio { child, .. } = self { let _ = child.kill(); let _ = child.wait(); }
    }
}

impl Conn {
    fn open(spec: &Spec) -> anyhow::Result<Conn> {
        let mut c = match spec {
            Spec::Stdio { command, args, env } => {
                // Windows starts `npx`/`uvx` through their .cmd shims, which only cmd.exe resolves.
                let mut p = if cfg!(windows) { let mut p = std::process::Command::new("cmd"); p.arg("/c").arg(command); p } else { std::process::Command::new(command) };
                let mut child = p.args(args).envs(env.iter().cloned()).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::null()).spawn().with_context(|| format!("start {command}"))?;
                let (stdin, out) = (child.stdin.take().unwrap(), child.stdout.take().unwrap());
                let (tx, rx) = std::sync::mpsc::channel();
                // Ends when the child closes stdout or the receiver is dropped.
                std::thread::spawn(move || { for l in BufReader::new(out).lines().map_while(Result::ok) { if tx.send(l).is_err() { break; } } });
                Conn::Stdio { child, stdin, rx }
            }
            Spec::Http { url, headers } => {
                let t = Some(timeout());
                let agent = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(30))).timeout_global(t).http_status_as_error(false).build().into();
                Conn::Http { agent, url: url.clone(), headers: headers.clone(), session: None }
            }
        };
        c.rpc(1, "initialize", json!({"protocolVersion": VERSION, "capabilities": {}, "clientInfo": {"name": "rustingstrata", "version": env!("CARGO_PKG_VERSION")}}))?;
        c.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}), None)?;
        Ok(c)
    }

    /// Sends one message. Over HTTP the answer to request `id` comes back in the response (JSON or an event stream).
    fn send(&mut self, m: &Value, id: Option<u64>) -> anyhow::Result<Option<Value>> {
        match self {
            Conn::Stdio { stdin, .. } => { writeln!(stdin, "{m}")?; stdin.flush()?; Ok(None) }
            Conn::Http { agent, url, headers, session } => {
                let mut r = agent.post(url.as_str()).header("Content-Type", "application/json").header("Accept", "application/json, text/event-stream")
                    .header("MCP-Protocol-Version", VERSION);
                for (k, v) in headers.iter() { r = r.header(k.as_str(), v.as_str()); }
                if let Some(s) = session.as_deref() { r = r.header("Mcp-Session-Id", s); }
                let resp = r.send(m.to_string()).with_context(|| format!("POST {url}"))?;
                let status = resp.status().as_u16();
                if let Some(s) = resp.headers().get("mcp-session-id").and_then(|s| s.to_str().ok()) { *session = Some(s.to_string()); }
                let stream = resp.headers().get("content-type").and_then(|c| c.to_str().ok()).is_some_and(|c| c.starts_with("text/event-stream"));
                if status >= 400 { bail!("{url} answered HTTP {status}"); }
                let Some(id) = id else { return Ok(None) };
                if !stream {
                    let mut b = String::new();
                    // ponytail: a 16 MB answer cap; raise it when a tool legitimately returns more.
                    resp.into_body().into_reader().take(16 << 20).read_to_string(&mut b)?;
                    return Ok(Some(serde_json::from_str(&b).context("MCP answer is not JSON")?));
                }
                for l in BufReader::new(resp.into_body().into_reader()).lines() {
                    let l = l?;
                    let Some(d) = l.strip_prefix("data:") else { continue };
                    if let Ok(v) = serde_json::from_str::<Value>(d.trim()) && v["id"] == id && v.get("method").is_none() { return Ok(Some(v)); }
                }
                bail!("{url}: the stream ended without an answer")
            }
        }
    }

    /// One request and its result. Server requests that arrive meanwhile are answered (`ping`) or refused.
    fn rpc(&mut self, id: u64, method: &str, params: Value) -> anyhow::Result<Value> {
        let direct = self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}), Some(id))?;
        let end = std::time::Instant::now() + timeout();
        let v = match direct {
            Some(v) => v,
            None => loop {
                let Conn::Stdio { rx, .. } = self else { unreachable!() };
                let l = match rx.recv_timeout(end.saturating_duration_since(std::time::Instant::now())) {
                    Ok(l) => l,
                    Err(RecvTimeoutError::Timeout) => bail!("no answer to {method} in {} s", timeout().as_secs()),
                    Err(RecvTimeoutError::Disconnected) => bail!("the server exited during {method}"),
                };
                // Servers may print other lines on stdout; anything that is not a JSON object is skipped.
                let Ok(v) = serde_json::from_str::<Value>(&l) else { continue };
                if v.get("method").is_some() {
                    if !v["id"].is_null() {
                        let a = if v["method"] == "ping" { json!({"jsonrpc": "2.0", "id": v["id"], "result": {}}) }
                            else { json!({"jsonrpc": "2.0", "id": v["id"], "error": {"code": -32601, "message": "not supported"}}) };
                        self.send(&a, None)?;
                    }
                    continue;
                }
                if v["id"] == id { break v; }
            },
        };
        if let Some(e) = v.get("error") { bail!("{}", e["message"].as_str().unwrap_or("MCP error")); }
        Ok(v["result"].clone())
    }
}

struct Server { spec: Spec, conn: Option<Conn>, next: u64, tools: Vec<Value>, error: Option<String> }

impl Server {
    /// Starts the server and reads its tool list, unless it is already up.
    fn up(&mut self) -> anyhow::Result<()> {
        if self.conn.is_some() { return Ok(()); }
        let mut c = Conn::open(&self.spec)?;
        let (mut tools, mut cursor) = (Vec::new(), None::<String>);
        loop {
            self.next += 1;
            let r = c.rpc(self.next, "tools/list", cursor.map_or(json!({}), |c| json!({"cursor": c})))?;
            tools.extend(r["tools"].as_array().into_iter().flatten().map(|t| json!({"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]})));
            cursor = r["nextCursor"].as_str().map(String::from);
            // ponytail: at most 1000 tools per server, so a server that always returns a cursor cannot loop forever.
            if cursor.is_none() || tools.len() >= 1000 { break; }
        }
        (self.conn, self.tools) = (Some(c), tools);
        Ok(())
    }
}

/// All servers of one config file. Each server has its own lock: a slow tool blocks only its own server (the name
/// is kept outside that lock so finding a server never waits for a running call).
pub struct Hub { path: PathBuf, servers: Mutex<Vec<(String, std::sync::Arc<Mutex<Server>>)>>, loaded: Mutex<bool>, error: Mutex<Option<String>> }

impl Hub {
    pub fn new(path: PathBuf) -> Hub { Hub { path, servers: Default::default(), loaded: Default::default(), error: Default::default() } }

    /// Reads the file on first use (and again on `reload`, which also stops the running servers).
    fn load(&self, reload: bool) {
        let mut loaded = self.loaded.lock().unwrap();
        if *loaded && !reload { return; }
        *loaded = true;
        let r = match std::fs::read_to_string(&self.path) {
            Ok(t) => parse(&t),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(anyhow!("{}: {e}", self.path.display())),
        };
        *self.error.lock().unwrap() = r.as_ref().err().map(|e| format!("{e:#}"));
        *self.servers.lock().unwrap() = r.unwrap_or_default().into_iter()
            .map(|(name, spec)| (name, std::sync::Arc::new(Mutex::new(Server { spec, conn: None, next: 1, tools: Vec::new(), error: None })))).collect();
    }

    /// The servers with their tools (starting those not yet up, in parallel) and, per server, why it failed.
    pub fn list(&self, reload: bool) -> Value {
        self.load(reload);
        let servers = self.servers.lock().unwrap().clone();
        std::thread::scope(|sc| for (_, s) in &servers { sc.spawn(move || { let mut s = s.lock().unwrap(); s.error = s.up().err().map(|e| format!("{e:#}")); }); });
        let list: Vec<_> = servers.iter().map(|(name, s)| { let s = s.lock().unwrap(); json!({"name": name, "error": s.error, "tools": s.tools}) }).collect();
        json!({"path": self.path, "error": *self.error.lock().unwrap(), "servers": list})
    }

    /// Calls `tool` on `server`. A protocol failure (or a timeout) is an error and drops the connection, so the next
    /// call starts the server again; a tool's own failure comes back as `is_error` for the model to read.
    pub fn call(&self, server: &str, tool: &str, args: Value) -> anyhow::Result<Value> {
        self.load(false);
        let s = self.servers.lock().unwrap().iter().find(|(n, _)| n == server).map(|(_, s)| s.clone()).with_context(|| format!("no MCP server {server:?}"))?;
        let mut s = s.lock().unwrap();
        s.up()?;
        s.next += 1;
        let id = s.next;
        let r = s.conn.as_mut().unwrap().rpc(id, "tools/call", json!({"name": tool, "arguments": args}));
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                // "unknown tool" and the like leave the connection usable; a timeout or an exit does not.
                let m = format!("{e:#}");
                if m.starts_with("no answer") || m.contains("exited") || m.contains("POST ") { s.conn = None; }
                return Err(e);
            }
        };
        let text: Vec<String> = r["content"].as_array().into_iter().flatten().map(|b| match b["type"].as_str() {
            Some("text") => b["text"].as_str().unwrap_or_default().to_string(),
            // ponytail: images and resources are not passed to the model; add when a vision model should see them.
            t => format!("[{}]", t.unwrap_or("content")),
        }).collect();
        Ok(json!({"content": text.join("\n"), "is_error": r["isError"] == true}))
    }
}
