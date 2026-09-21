// Runs a `wasm32-wasip1` binary under the WASI support of Node 22 or newer,
// so that
// `cargo test --target wasm32-wasip1` and
// `cargo bench -p sha1dc-bench --target wasm32-wasip1` work with nothing
// installed beyond Node itself. `.cargo/config.toml` makes this script the
// runner of that target, and hands it the path of the module followed by the
// arguments of the harness.
//
// Wasmtime is an alternative that needs no Node, and reports what a different
// engine makes of the same module. Make it the runner instead:
//
//     runner = ["wasmtime", "--dir", ".", "--dir", "target", "--"]

import { mkdirSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WASI } from 'node:wasi';

// The bootstrap in `.cargo/config.toml` imports this script rather than
// running it as the entry point, which leaves the arguments one place
// earlier than they are when it is started by name.
const argv = process.argv.slice(1);
const [modulePath, ...args] =
  argv[0] === fileURLToPath(import.meta.url) ? argv.slice(1) : argv;

if (modulePath === undefined) {
  console.error('usage: wasm-runner.mjs <module.wasm> [args...]');
  process.exit(2);
}

// Criterion looks for the target directory with `cargo metadata`, which a
// module cannot run, and would fall back to a `target` of its own next to
// whichever directory it runs in. Hand it one under the directory Cargo
// already built into, where the `--save-baseline` of one run is still there
// for the `--baseline` of the next. Keeping it per-target also keeps these
// estimates apart from the native ones, which are not comparable.
const criterionHome = resolve(dirname(resolve(modulePath)), '..', '..', 'criterion');
mkdirSync(criterionHome, { recursive: true });

const wasi = new WASI({
  version: 'preview1',
  args: [modulePath, ...args],
  env: { ...process.env, CRITERION_HOME: criterionHome },
  // A module reaches only what it is given: the directory it runs in, and
  // the one its estimates go to. Cargo runs it in the directory of the
  // package, which is the `CARGO_MANIFEST_DIR` the tests join their data
  // paths onto, so that directory is granted under its own name as well.
  // Without it those paths resolve against no preopen and the tests that
  // read a file skip themselves instead of failing.
  preopens: {
    '.': process.cwd(),
    [process.cwd()]: process.cwd(),
    [criterionHome]: criterionHome,
  },
  returnOnExit: true,
});

const module = await WebAssembly.compile(await readFile(modulePath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());

process.exit(wasi.start(instance));
