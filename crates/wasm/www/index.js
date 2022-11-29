import init, { NeuralNet } from "./wasm/wasm.js";

await init();

const neuralNet = NeuralNet.new();

const canvas = document.getElementById("sheet");
canvas.height = canvas.width = 28 * 20;
const g = canvas.getContext("2d");
g.lineJoin = "round";
g.lineWidth = 10;
g.strokeStyle = "white";
g.filter = "blur(12px)";

const relPos = (pt) => [
    pt.pageX - canvas.offsetLeft,
    pt.pageY - canvas.offsetTop,
  ],
  drawStart = (pt) => {
    g.beginPath();
    g.moveTo.apply(g, pt);
    g.stroke();
  },
  drawMove = (pt) => {
    g.lineTo.apply(g, pt);
    g.stroke();
  },
  pointerDown = (e) => drawStart(relPos(e.touches ? e.touches[0] : e)),
  pointerMove = (e) => drawMove(relPos(e.touches ? e.touches[0] : e)),
  draw = (method, move, stop) => (e) => {
    if (method == "add") pointerDown(e);
    canvas[method + "EventListener"](move, pointerMove);
    canvas[method + "EventListener"](stop, g.closePath);
  };

canvas.addEventListener("mousedown", draw("add", "mousemove", "mouseup"));
canvas.addEventListener("touchstart", draw("add", "touchmove", "touchend"));
canvas.addEventListener("mouseup", draw("remove", "mousemove", "mouseup"));
canvas.addEventListener("touchend", draw("remove", "touchmove", "touchend"));

document.getElementById("clear").addEventListener("click", () => {
  g.clearRect(0, 0, canvas.width, canvas.height);
});

const grid = document.getElementById("grid");
const output = document.getElementById("output");
const breakdownRows = document.querySelectorAll("td.confidence");

document.getElementById("submit").addEventListener("click", () => {
  const newCanvas = document.createElement("canvas");
  const ctx = newCanvas.getContext("2d");
  ctx.drawImage(canvas, 0, 0, 28, 28);

  const rawImageData = ctx.getImageData(0, 0, 28, 28).data;

  if (rawImageData.length !== 28 * 28 * 4) {
    throw new Error();
  }

  const imageData = [];

  const frag = document.createDocumentFragment();

  for (let i = 0; i < rawImageData.length; i += 4) {
    const r = rawImageData[i];
    const g = rawImageData[i + 1];
    const b = rawImageData[i + 2];

    const average = Math.round((r + g + b) / 3);

    const pixel = document.createElement("div");
    pixel.classList.add("pixel");
    pixel.style.background = `rgb(${average},${average},${average})`;

    frag.appendChild(pixel);

    imageData.push(average);
  }

  grid.innerHTML = "";
  grid.appendChild(frag);

  const breakdown = new Float32Array(10);

  const prediction = neuralNet.classify(imageData, breakdown);
  output.textContent = prediction;

  for (let i = 0; i < breakdownRows.length; i++) {
    const row = breakdownRows[i];
    const confidence = breakdown[i] * 100;
    row.textContent = `${confidence.toFixed(2)}%`;
  }
});
