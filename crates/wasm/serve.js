// Import http and fs
const fs = require("fs");
const http = require("http");
const path = require("path");

// Creating Server
http
  .createServer((req, res) => {
    // Reading file
    let encoding = "utf8";
    if (req.url.endsWith(".wasm")) {
      encoding = null;
    }
    // console.log(req.url);
    let filePath = path.join(__dirname, "/www", req.url);
    // console.log(filePath);

    fs.readFile(filePath, { encoding }, function (err, filedata) {
      if (err) {
        console.error(err);
        // Handling error
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
      // console.log(contentType);
      res.setHeader("Content-Type", contentType);
      // serving file to the server
      res.writeHead(200);
      res.end(filedata);
    });
    // console.log("\n");
  })
  .listen(8080, "localhost");
