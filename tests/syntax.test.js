const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const dirs = ["docs", "docs/kit"];
let files = [];
for (const d of dirs) {
  for (const f of fs.readdirSync(path.join(__dirname, "..", d))) {
    if (f.endsWith(".js")) files.push(path.join(__dirname, "..", d, f));
  }
}
let bad = 0;
for (const f of files) {
  try {
    execFileSync(process.execPath, ["--input-type=module", "--check"], { input: fs.readFileSync(f), stdio: ["pipe", "pipe", "pipe"] });
    console.log("ok   " + path.relative(process.cwd(), f));
  } catch (e) {
    bad++;
    console.error("FAIL " + path.relative(process.cwd(), f) + "\n" + String(e.stderr).split("\n").slice(0, 4).join("\n"));
  }
}
if (bad) { console.error(bad + " file(s) failed ESM parse"); process.exit(1); }
console.log("\nALL SYNTAX TESTS PASSED (" + files.length + " files)");
