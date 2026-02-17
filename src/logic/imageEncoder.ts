import { layerMap1 } from "../matrix/imageEncoder/layer1/layer1";
import { layerMap2 } from "../matrix/imageEncoder/layer1/layer2";
import { linearLayer } from "../matrix/imageEncoder/layer1/linearLayer"; // number[48][1250]
import { linearBias } from "../matrix/imageEncoder/layer1/linearBias"; // number[48]

const buildImage = (imageData: number[]): number[][] => {
  const image: number[][] = [];
  for (let j = 0; j < 28; j++) {
    const start = j * 28;
    image.push(imageData.slice(start, start + 28));
  }
  return image;
};

const applyReLU = (featureMaps: number[][][]): number[][][] => {
  return featureMaps.map(map =>
    map.map(row =>
      row.map(value => Math.max(0, value))
    )
  );
};

const masking = (
  image: number[][],
  masks: { index: number; map: number[][] }[]
): number[][][] => {
  const sortedMasks = [...masks].sort((a, b) => a.index - b.index);
  const kernelSize = sortedMasks[0].map.length;
  const imageSize = image.length;
  const outputSize = imageSize - kernelSize + 1;

  const result: number[][][] = [];

  for (const mask of sortedMasks) {
    const kernel = mask.map;
    const featureMap: number[][] = [];

    for (let i = 0; i < outputSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < outputSize; j++) {
        let sum = 0;
        for (let ki = 0; ki < kernelSize; ki++) {
          for (let kj = 0; kj < kernelSize; kj++) {
            sum += image[i + ki][j + kj] * kernel[ki][kj];
          }
        }
        row.push(sum);
      }
      featureMap.push(row);
    }
    result.push(featureMap);
  }

  return result;
};

const masking3d = (
  input: number[][][],
  kernels: number[][][][]
): number[][][] => {
  const inChannels = input.length;
  const height = input[0].length;
  const width = input[0][0].length;
  const outChannels = kernels.length;
  const kernelSize = kernels[0][0].length; // предполагаем квадратные ядра

  const outHeight = height - kernelSize + 1;
  const outWidth = width - kernelSize + 1;

  const result: number[][][] = [];

  for (let outC = 0; outC < outChannels; outC++) {
    const kernelSet = kernels[outC];
    const featureMap: number[][] = [];

    for (let i = 0; i < outHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < outWidth; j++) {
        let sum = 0;
        for (let c = 0; c < inChannels; c++) {
          for (let ki = 0; ki < kernelSize; ki++) {
            for (let kj = 0; kj < kernelSize; kj++) {
              sum += input[c][i + ki][j + kj] * kernelSet[c][ki][kj];
            }
          }
        }
        row.push(sum);
      }
      featureMap.push(row);
    }
    result.push(featureMap);
  }

  return result;
};

// Max pooling с окном 2x2 и шагом 2.
const maxPooling2x2 = (featureMaps: number[][][]): number[][][] => {
  const numMaps = featureMaps.length;
  const inputSize = featureMaps[0].length; // квадратные карты
  const outputSize = Math.floor(inputSize / 2);

  const pooled: number[][][] = [];

  for (let m = 0; m < numMaps; m++) {
    const map = featureMaps[m];
    const outMap: number[][] = [];

    for (let i = 0; i < outputSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < outputSize; j++) {
        let maxVal = -Infinity;
        for (let di = 0; di < 2; di++) {
          for (let dj = 0; dj < 2; dj++) {
            const val = map[i * 2 + di][j * 2 + dj];
            if (val > maxVal) maxVal = val;
          }
        }
        row.push(maxVal);
      }
      outMap.push(row);
    }
    pooled.push(outMap);
  }
  return pooled;
};


// Преобразует трёхмерный массив карт признаков в одномерный вектор.
const flatten = (featureMaps: number[][][]): number[] => {
  const flat: number[] = [];
  for (const map of featureMaps) {
    for (const row of map) {
      flat.push(...row);
    }
  }
  return flat;
};

const linearForward = (input: number[], weights: number[][], bias: number[]): number[] => {
  const outputSize = weights.length; // 48
  const output = new Array(outputSize);

  for (let i = 0; i < outputSize; i++) {
    let sum = bias[i];
    const row = weights[i];
    // Скалярное произведение входного вектора и строки весов
    for (let j = 0; j < input.length; j++) {
      sum += input[j] * row[j];
    }
    output[i] = sum;
  }
  return output;
};

export const imageEncoder = (imageData: number[]): number[] => {
  // 1. Нормализация в [0, 1]
  const normalized = imageData.map(p => p / 255.0);

  // 2. Построение матрицы 28x28
  const imageArr = buildImage(normalized);

  // 3. Первый свёрточный слой + ReLU + MaxPooling
  const conv1 = masking(imageArr, layerMap1);
  const activated1 = applyReLU(conv1);
  const pool1 = maxPooling2x2(activated1);

  // 4. Второй свёрточный слой + ReLU + MaxPooling
  const conv2 = masking3d(pool1, layerMap2);
  const activated2 = applyReLU(conv2);
  const pool2 = maxPooling2x2(activated2);

  // 5. Развёртка в вектор размером 1250
  const encoded = flatten(pool2);

  // 6. Полносвязный слой (проекция в 48)
  const output = linearForward(encoded, linearLayer, linearBias);

  return output;
};