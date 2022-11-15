import fs from "fs";
import http from "http";
import { existsSync, mkdirSync } from "fs";
import { tmpdir } from "os";
import { sep } from "path";
import { createGunzip } from "zlib";
import { createHash } from "crypto";

const mnistTmpDir = `${tmpdir()}${sep}mnist-dataset`;

export const gunzipUrl = (url) =>
  Promise.resolve([])
    .then(
      (buf) =>
        new Promise((resolve, reject) =>
          http
            .get(url, (res) =>
              res
                .pipe(createGunzip())
                .on("data", (data) => buf.push(data))
                .on("end", () => resolve(buf))
                .on("error", reject)
            )
            .on("error", reject)
        )
    )
    .then((arr) => [].concat(...arr))
    .then((arr) => Buffer.concat(arr));

export const getTempPath = (key) => {
  if (!existsSync(mnistTmpDir)) mkdirSync(mnistTmpDir);

  if (typeof key === "string") {
    return `${mnistTmpDir}${sep}${createHash("md5").update(key).digest("hex")}`;
  }

  throw new Error(`Expected string key, encountered ${key}.`);
};

const magicNumbers = Object.freeze([2051, 2049]);

const [imagesMagicNumber, labelsMagicNumber] = magicNumbers;

const images = (buf) =>
  Promise.resolve().then(() => {
    const [magicNumber, numberOfImages, numberOfRows, numberOfColumns] = [
      buf.readInt32BE(0),
      buf.readInt32BE(4),
      buf.readInt32BE(8),
      buf.readInt32BE(12),
    ];
    const { data } = buf.slice(16).toJSON();
    const pixels = numberOfRows * numberOfColumns;
    return [...Array(numberOfImages)].map((_, i) =>
      data.slice(i * pixels, (i + 1) * pixels)
    );
  });

const labels = (buf) =>
  Promise.resolve().then(() => {
    const [magicNumber, numberOfLabels] = [
      buf.readInt32BE(0),
      buf.readInt32BE(4),
    ];
    const { data } = buf.slice(8).toJSON();
    return [...Array(numberOfLabels)].map((_, i) => data.slice(i, i + 1));
  });

const parse = (buf) =>
  Promise.resolve().then(() => {
    const magicNumber = buf.readInt32BE(0);
    if (magicNumber === imagesMagicNumber) {
      return images(buf);
    } else if (magicNumber === labelsMagicNumber) {
      return labels(buf);
    }
    return Promise.reject(
      new Error(
        `Expected one of magic numbers ${magicNumbers}, encountered ${magicNumber}.`
      )
    );
  });

const maybeDownload = (url) =>
  Promise.resolve().then(() => {
    const path = getTempPath(url);
    if (!fs.existsSync(path)) {
      return gunzipUrl(url)
        .then((data) => fs.promises.writeFile(path, data))
        .then(() => path);
    }
    return path;
  });

export const trainingImagesUrl =
  "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
export const trainingLabelsUrl =
  "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
export const testImagesUrl =
  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
export const testLabelsUrl =
  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

export const cache = (url) =>
  maybeDownload(url)
    .then((path) => fs.promises.readFile(path))
    .then(parse);

(async () => {
  // const trainingImages = await cache(trainingImagesUrl); // [[Number]] (i.e. [[0, 0, 0, ...], [0, 0, 0, ...], ...])
  const trainingLabels = await cache(trainingLabelsUrl); // [[Number]] (i.e. [[7], [2], ...])

  console.log(trainingLabels.slice(0, 5));

  // const toString = (e) =>
  //   [...Array(28)]
  //     .map((_, i) =>
  //       [...Array(28)].map((_, j) => (e[i * 28 + j] > 128 ? "#" : " ")).join("")
  //     )
  //     .join("\n");

  // const [number7] = trainingImages;
  // console.log(toString(number7));
})();
// import fs from "fs";
// import http from "http";
// import path from "path";
// import { existsSync, mkdirSync } from "fs";
// import { tmpdir } from "os";
// import { sep } from "path";
// import { createGunzip } from "zlib";
// import { createHash } from "crypto";

// const mnistTmpDir = `${tmpdir()}${sep}mnist-dataset`;

// export const gunzipUrl = (url) =>
//   Promise.resolve([])
//     .then(
//       (buf) =>
//         new Promise((resolve, reject) =>
//           http
//             .get(url, (res) =>
//               res
//                 .pipe(createGunzip())
//                 .on("data", (data) => buf.push(data))
//                 .on("end", () => resolve(buf))
//                 .on("error", reject)
//             )
//             .on("error", reject)
//         )
//     )
//     .then((arr) => [].concat(...arr))
//     .then((arr) => Buffer.concat(arr));

// export const getTempPath = (key) => {
//   if (!existsSync(mnistTmpDir)) mkdirSync(mnistTmpDir);

//   if (typeof key === "string") {
//     return `${mnistTmpDir}${sep}${createHash("md5").update(key).digest("hex")}`;
//   }

//   throw new Error(`Expected string key, encountered ${key}.`);
// };

// const magicNumbers = Object.freeze([2051, 2049]);

// const [imagesMagicNumber, labelsMagicNumber] = magicNumbers;

// const images = (buf) =>
//   Promise.resolve().then(() => {
//     const [magicNumber, numberOfImages, numberOfRows, numberOfColumns] = [
//       buf.readInt32BE(0),
//       buf.readInt32BE(4),
//       buf.readInt32BE(8),
//       buf.readInt32BE(12),
//     ];
//     const { data } = buf.slice(16).toJSON();
//     const pixels = numberOfRows * numberOfColumns;
//     return [...Array(numberOfImages)].map((_, i) =>
//       data.slice(i * pixels, (i + 1) * pixels)
//     );
//   });

// const labels = (buf) =>
//   Promise.resolve().then(() => {
//     const [magicNumber, numberOfLabels] = [
//       buf.readInt32BE(0),
//       buf.readInt32BE(4),
//     ];
//     const { data } = buf.slice(8).toJSON();
//     return [...Array(numberOfLabels)].map((_, i) => data.slice(i, i + 1));
//   });

// const parse = (buf) =>
//   Promise.resolve().then(() => {
//     const magicNumber = buf.readInt32BE(0);
//     if (magicNumber === imagesMagicNumber) {
//       return images(buf);
//     } else if (magicNumber === labelsMagicNumber) {
//       return labels(buf);
//     }
//     return Promise.reject(
//       new Error(
//         `Expected one of magic numbers ${magicNumbers}, encountered ${magicNumber}.`
//       )
//     );
//   });

// const download = (url) =>
//   gunzipUrl(url).then((data) =>
//     fs.promises.writeFile(path.parse(url).name, data)
//   );

// export const trainingImagesUrl =
//   "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
// export const trainingLabelsUrl =
//   "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
// export const testImagesUrl =
//   "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
// export const testLabelsUrl =
//   "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

// // export const cache = (url) =>
// //   maybeDownload(url)
// //     .then((path) => fs.promises.readFile(path))
// //     .then(parse);

// (async () => {
//   await download(trainingImagesUrl);
//   await download(trainingLabelsUrl);
//   await download(testImagesUrl);
//   await download(testLabelsUrl);
//   // const trainingImages = await cache(trainingImagesUrl); // [[Number]] (i.e. [[0, 0, 0, ...], [0, 0, 0, ...], ...])
//   // const trainingLabels = await cache(trainingLabelsUrl); // [[Number]] (i.e. [[7], [2], ...])

//   // const toString = (e) =>
//   //   [...Array(28)]
//   //     .map((_, i) =>
//   //       [...Array(28)].map((_, j) => (e[i * 28 + j] > 128 ? "#" : " ")).join("")
//   //     )
//   //     .join("\n");

//   // const [number7] = trainingImages;
//   // console.log(toString(number7));
// })();
