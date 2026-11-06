## Issue

Closes #

## Cosa cambia

<!-- punti concreti, con file:riga -->

## Come è stato verificato

```sh
# comandi eseguiti
```

## Checklist

- [ ] scope = contratto della issue
- [ ] `cargo build --release` verde
- [ ] `cargo test --release` verde
- [ ] test GPU eseguiti (se previsti) e output nel Dev report
- [ ] nessuna regressione misurata
- [ ] modifiche Volta dietro `__CUDA_ARCH__` (se CUDA)
- [ ] SPIR-V rigenerati con `tools/gen_spirv.sh` (se Vulkan)
- [ ] niente segreti, modelli o binari
