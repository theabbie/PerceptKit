import fs from "node:fs";

const files = process.argv.slice(2);

function strip(src) {
  const out = [];
  let i = 0;
  const n = src.length;
  let state = "code";
  let quote = null;
  let braceDepth = 0;
  const tplDepth = [];
  let regexClass = false;

  const lastSig = () => {
    for (let j = out.length - 1; j >= 0; j--) {
      const c = out[j];
      if (c === " " || c === "\t" || c === "\n" || c === "\r") continue;
      return c;
    }
    return "";
  };

  const startsRegex = () => {
    const c = lastSig();
    if (c === "" || "([{=,:;!&|?+-*%^~<>".includes(c)) return true;
    const m = out.join("").match(/([A-Za-z_$][\w$]*)$/);
    if (m && ["return", "typeof", "case", "delete", "void", "new", "in", "of", "instanceof", "throw", "else", "do", "yield", "await"].includes(m[1])) return true;
    return false;
  };

  while (i < n) {
    const c = src[i];
    const d = src[i + 1];

    if (state === "code") {
      if (c === "/" && d === "/") {
        while (i < n && src[i] !== "\n") i++;
        continue;
      }
      if (c === "/" && d === "*") {
        i += 2;
        while (i < n && !(src[i] === "*" && src[i + 1] === "/")) {
          if (src[i] === "\n") out.push("\n");
          i++;
        }
        i += 2;
        continue;
      }
      if (c === "'") { state = "str"; quote = "'"; out.push(c); i++; continue; }
      if (c === '"') { state = "str"; quote = '"'; out.push(c); i++; continue; }
      if (c === "`") { state = "tpl"; out.push(c); i++; continue; }
      if (c === "/" && startsRegex()) {
        state = "regex"; regexClass = false; out.push(c); i++; continue;
      }
      if (c === "{") { braceDepth++; out.push(c); i++; continue; }
      if (c === "}") {
        if (tplDepth.length && tplDepth[tplDepth.length - 1] === braceDepth) {
          tplDepth.pop(); braceDepth--; out.push(c); i++; state = "tpl"; continue;
        }
        braceDepth--; out.push(c); i++; continue;
      }
      out.push(c); i++;
    } else if (state === "str") {
      if (c === "\\") { out.push(c); out.push(src[i + 1] ?? ""); i += 2; continue; }
      out.push(c);
      if (c === quote) state = "code";
      i++;
    } else if (state === "tpl") {
      if (c === "\\") { out.push(c); out.push(src[i + 1] ?? ""); i += 2; continue; }
      if (c === "`") { state = "code"; out.push(c); i++; continue; }
      if (c === "$" && d === "{") {
        out.push("${"); tplDepth.push(braceDepth); state = "code"; i += 2; continue;
      }
      out.push(c); i++;
    } else if (state === "regex") {
      if (c === "\\") { out.push(c); out.push(src[i + 1] ?? ""); i += 2; continue; }
      if (c === "[") regexClass = true;
      if (c === "]") regexClass = false;
      out.push(c);
      if (c === "/" && !regexClass) state = "code";
      i++;
    }
  }
  return out.join("");
}

for (const f of files) {
  const src = fs.readFileSync(f, "utf8");
  fs.writeFileSync(f, strip(src));
  console.log("stripped", f);
}
