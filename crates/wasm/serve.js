const fs = require("fs");
const http = require("http");
const path = require("path");

http
  .createServer((req, res) => {
    let encoding = "utf8";
    if (req.url.endsWith(".wasm")) {
      encoding = null;
    }
    let filePath = path.join(__dirname, "/www", req.url);

    fs.readFile(filePath, { encoding }, function (err, filedata) {
      if (err) {
        console.error(err);
        res.writeHead(404);
        res.end(JSON.stringify(err));
        return;
      }
      let contentType = "text";
      if (req.url.endsWith(".js")) {
        contentType = "text/javascript";
      } else if (req.url.endsWith(".html")) {
        contentType = "text/html";
      } else if (req.url.endsWith(".css")) {
        contentType = "text/css";
      } else if (req.url.endsWith(".wasm")) {
        contentType = "application/wasm";
      }
      res.setHeader("Content-Type", contentType);
      res.writeHead(200);
      res.end(filedata);
    });
  })
  .listen(8080, "localhost");
