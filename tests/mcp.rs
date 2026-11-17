use rusting_strata::mcp::{Hub, Spec, parse};
use serde_json::json;

const ECHO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mcp_echo.py");

fn python() -> bool { std::process::Command::new("python3").arg("--version").output().is_ok_and(|o| o.status.success()) }

/// A hub over a config file with the fake server and one command that does not exist.
fn hub(name: &str) -> Hub {
    let f = std::env::temp_dir().join(format!("strata-mcp-{name}-{}.json", std::process::id()));
    std::fs::write(&f, json!({"mcpServers": {"echo": {"command": "python3", "args": [ECHO]}, "gone": {"command": "strata-no-such-command"}}}).to_string()).unwrap();
    Hub::new(f)
}

#[test]
fn parses_the_config() {
    let s = parse(r#"{"mcpServers": {"fs": {"command": "npx", "args": ["-y", "x"], "env": {"A": "b"}}, "web": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer k"}}}}"#).unwrap();
    assert_eq!(s.len(), 2);
    assert!(matches!(&s[0], (n, Spec::Stdio { command, args, env }) if n == "fs" && command == "npx" && args == &["-y", "x"] && env == &[("A".to_string(), "b".to_string())]));
    assert!(matches!(&s[1], (n, Spec::Http { url, headers }) if n == "web" && url == "https://example.com/mcp" && headers.len() == 1));
    assert!(parse("{}").unwrap().is_empty());
    assert!(parse(r#"{"mcpServers": {"x": {}}}"#).is_err(), "neither command nor url");
    assert!(parse(r#"{"mcpServers": {"x": {"url": "file:///etc/passwd"}}}"#).is_err(), "only http and https");
    assert!(parse("not json").is_err());
}

#[test]
fn lists_and_calls_tools() {
    if !python() { eprintln!("skip: no python3"); return; }
    let h = hub("list");
    let v = h.list(false);
    let echo = &v["servers"][0];
    assert_eq!(echo["name"], "echo");
    assert_eq!(echo["error"], json!(null));
    assert_eq!(echo["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect::<Vec<_>>(), ["echo", "fail", "hang"]);
    assert_eq!(echo["tools"][0]["inputSchema"]["required"], json!(["text"]));
    // A server that does not start is listed with its error and does not hide the others.
    assert_eq!(v["servers"][1]["name"], "gone");
    assert!(v["servers"][1]["error"].as_str().unwrap().contains("strata-no-such-command"));
    assert_eq!(v["servers"][1]["tools"], json!([]));

    // Text blocks join; a block that is not text becomes a placeholder.
    assert_eq!(h.call("echo", "echo", json!({"text": "hi"})).unwrap(), json!({"content": "hi\n[image]", "is_error": false}));
    assert_eq!(h.call("echo", "fail", json!({})).unwrap(), json!({"content": "boom", "is_error": true}));
    assert!(h.call("echo", "nope", json!({})).unwrap_err().to_string().contains("unknown tool"));
    assert!(h.call("nope", "echo", json!({})).unwrap_err().to_string().contains("no MCP server"));
}

#[test]
fn a_hung_call_times_out_and_the_server_restarts() {
    if !python() { eprintln!("skip: no python3"); return; }
    // SAFETY: set before the hub starts any thread that reads it.
    unsafe { std::env::set_var("STRATA_MCP_TIMEOUT", "1") };
    let h = hub("hang");
    let t = std::time::Instant::now();
    assert!(h.call("echo", "hang", json!({})).unwrap_err().to_string().contains("no answer"));
    assert!(t.elapsed().as_secs() < 10);
    // The stuck process was dropped; the next call starts a new one.
    assert_eq!(h.call("echo", "echo", json!({"text": "again"})).unwrap()["content"], "again\n[image]");
}

#[test]
fn a_missing_file_is_an_empty_list() {
    let v = Hub::new(std::env::temp_dir().join("strata-mcp-no-such-file.json")).list(false);
    assert_eq!(v["servers"], json!([]));
    assert!(v["path"].as_str().unwrap().ends_with("strata-mcp-no-such-file.json"));
}

/// The streamable-HTTP transport against a fake server on a local port: a JSON answer, an event-stream answer (with a
/// server request before the result), the session id sent back on later requests, and 202 for the notification.
#[test]
fn http_server_lists_and_calls() {
    use std::io::{BufRead, BufReader, Read, Write};
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/mcp", l.local_addr().unwrap());
    std::thread::spawn(move || for c in l.incoming() {
        let mut c = c.unwrap();
        let mut r = BufReader::new(c.try_clone().unwrap());
        let (mut len, mut session, mut auth) = (0, false, false);
        loop {
            let mut h = String::new();
            r.read_line(&mut h).unwrap();
            let h = h.trim().to_ascii_lowercase();
            if h.is_empty() { break; }
            if let Some(n) = h.strip_prefix("content-length:") { len = n.trim().parse().unwrap(); }
            session |= h == "mcp-session-id: s1";
            auth |= h == "authorization: bearer k";
        }
        let mut b = vec![0; len];
        r.read_exact(&mut b).unwrap();
        let m: serde_json::Value = serde_json::from_slice(&b).unwrap();
        let (id, method) = (&m["id"], m["method"].as_str().unwrap());
        let mut answer = |status: &str, head: &str, body: String| write!(c, "HTTP/1.1 {status}\r\n{head}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        let js = "Content-Type: application/json\r\n";
        match method {
            _ if !auth => answer("401 Unauthorized", "", String::new()),
            "initialize" => answer("200 OK", &format!("{js}Mcp-Session-Id: s1\r\n"), json!({"jsonrpc": "2.0", "id": id, "result": {"protocolVersion": "2025-06-18", "capabilities": {}}}).to_string()),
            _ if !session => answer("400 Bad Request", "", String::new()),
            "notifications/initialized" => answer("202 Accepted", "", String::new()),
            "tools/list" => answer("200 OK", js, json!({"jsonrpc": "2.0", "id": id, "result": {"tools": [{"name": "up", "description": "Upper case", "inputSchema": {"type": "object"}}]}}).to_string()),
            _ => answer("200 OK", "Content-Type: text/event-stream\r\n", format!("event: message\ndata: {}\n\ndata: {}\n\n",
                json!({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"}),
                json!({"jsonrpc": "2.0", "id": id, "result": {"content": [{"type": "text", "text": m["params"]["arguments"]["text"].as_str().unwrap().to_uppercase()}]}}))),
        }
    });
    let f = std::env::temp_dir().join(format!("strata-mcp-http-{}.json", std::process::id()));
    std::fs::write(&f, json!({"mcpServers": {"web": {"url": url, "headers": {"Authorization": "Bearer k"}}}}).to_string()).unwrap();
    let h = Hub::new(f);
    let v = h.list(false);
    assert_eq!(v["servers"][0]["error"], json!(null), "{v}");
    assert_eq!(v["servers"][0]["tools"][0]["name"], "up");
    assert_eq!(h.call("web", "up", json!({"text": "hi"})).unwrap(), json!({"content": "HI", "is_error": false}));
}
