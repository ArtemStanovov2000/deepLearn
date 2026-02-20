import { vocab } from "../matrix/textEncoder/vocab/vocab";

import { embeddingWeight } from "../matrix/textEncoder/embeddings/embeddingWeight"; // number[47][48]
import { posEncoderPe } from "../matrix/textEncoder/embeddings/posEncoderPe"; // number[64][48]

import { transformerLayers0Linear1Bias } from "../matrix/textEncoder/layer_1/transformerLayers0Linear1Bias"; // number[192]
import { transformerLayers0Linear1Weight } from "../matrix/textEncoder/layer_1/transformerLayers0Linear1Weight"; // number[192][48]
import { transformerLayers0Linear2Bias } from "../matrix/textEncoder/layer_1/transformerLayers0Linear2Bias"; // number[48]
import { transformerLayers0Linear2Weight } from "../matrix/textEncoder/layer_1/transformerLayers0Linear2Weight"; // number[48][192]
import { transformerLayers0Norm1Bias } from "../matrix/textEncoder/layer_1/transformerLayers0Norm1Bias"; // number[48]
import { transformerLayers0Norm1Weight } from "../matrix/textEncoder/layer_1/transformerLayers0Norm1Weight"; // number[48]
import { transformerLayers0Norm2Bias } from "../matrix/textEncoder/layer_1/transformerLayers0Norm2Bias"; // number[48]
import { transformerLayers0Norm2Weight } from "../matrix/textEncoder/layer_1/transformerLayers0Norm2Weight"; // number[48]
import { transformerLayers0Self_attnIn_proj_bias } from "../matrix/textEncoder/layer_1/transformerLayers0Self_attnIn_proj_bias"; // number[144]
import { transformerLayers0Self_attnIn_proj_weight } from "../matrix/textEncoder/layer_1/transformerLayers0Self_attnIn_proj_weight"; // number[144][48]
import { transformerLayers0Self_attnOut_projBias } from "../matrix/textEncoder/layer_1/transformerLayers0Self_attnOut_projBias"; // number[48]
import { transformerLayers0Self_attnOut_projWeight } from "../matrix/textEncoder/layer_1/transformerLayers0Self_attnOut_projWeight"; // number[44][48]

import { transformerLayers1Linear1Bias } from "../matrix/textEncoder/layer_2/transformerLayers1Linear1Bias"; // number[192]
import { transformerLayers1Linear1Weight } from "../matrix/textEncoder/layer_2/transformerLayers1Linear1Weight"; // number[192][48]
import { transformerLayers1Linear2Bias } from "../matrix/textEncoder/layer_2/transformerLayers1Linear2Bias"; // number[48]
import { transformerLayers1Linear2Weight } from "../matrix/textEncoder/layer_2/transformerLayers1Linear2Weight"; // number[48][192]
import { transformerLayers1Norm1Bias } from "../matrix/textEncoder/layer_2/transformerLayers1Norm1Bias"; // number[48]
import { transformerLayers1Norm1Weight } from "../matrix/textEncoder/layer_2/transformerLayers1Norm1Weight"; // number[48]
import { transformerLayers1Norm2Bias } from "../matrix/textEncoder/layer_2/transformerLayers1Norm2Bias"; // number[48]
import { transformerLayers1Norm2Weight } from "../matrix/textEncoder/layer_2/transformerLayers1Norm2Weight"; // number[48]
import { transformerLayers1Self_attnIn_proj_bias } from "../matrix/textEncoder/layer_2/transformerLayers1Self_attnIn_proj_bias"; // number[144]
import { transformerLayers1Self_attnIn_proj_weight } from "../matrix/textEncoder/layer_2/transformerLayers1Self_attnIn_proj_weight"; // number[144][48]
import { transformerLayers1Self_attnOut_projBias } from "../matrix/textEncoder/layer_2/transformerLayers1Self_attnOut_projBias"; // number[48]
import { transformerLayers1Self_attnOut_projWeight } from "../matrix/textEncoder/layer_2/transformerLayers1Self_attnOut_projWeight"; // number[44][48]

import { transformerLayers2Linear1Bias } from "../matrix/textEncoder/layer_3/transformerLayers2Linear1Bias"; // number[192]
import { transformerLayers2Linear1Weight } from "../matrix/textEncoder/layer_3/transformerLayers2Linear1Weight"; // number[192][48]
import { transformerLayers2Linear2Bias } from "../matrix/textEncoder/layer_3/transformerLayers2Linear2Bias"; // number[48]
import { transformerLayers2Linear2Weight } from "../matrix/textEncoder/layer_3/transformerLayers2Linear2Weight"; // number[48][192]
import { transformerLayers2Norm1Bias } from "../matrix/textEncoder/layer_3/transformerLayers2Norm1Bias"; // number[48]
import { transformerLayers2Norm1Weight } from "../matrix/textEncoder/layer_3/transformerLayers2Norm1Weight"; // number[48]
import { transformerLayers2Norm2Bias } from "../matrix/textEncoder/layer_3/transformerLayers2Norm2Bias"; // number[48]
import { transformerLayers2Norm2Weight } from "../matrix/textEncoder/layer_3/transformerLayers2Norm2Weight"; // number[48]
import { transformerLayers2Self_attnIn_proj_bias } from "../matrix/textEncoder/layer_3/transformerLayers2Self_attnIn_proj_bias"; // number[144]
import { transformerLayers2Self_attnIn_proj_weight } from "../matrix/textEncoder/layer_3/transformerLayers2Self_attnIn_proj_weight"; // number[144][48]
import { transformerLayers2Self_attnOut_projBias } from "../matrix/textEncoder/layer_3/transformerLayers2Self_attnOut_projBias"; // number[48]
import { transformerLayers2Self_attnOut_projWeight } from "../matrix/textEncoder/layer_3/transformerLayers2Self_attnOut_projWeight"; // number[44][48]

import { normBias } from "../matrix/textEncoder/layer_3/normBias"; // number[48]
import { normWeight } from "../matrix/textEncoder/layer_3/normWeight"; // number[48]


// Константы (обновлены в соответствии с вашим vocab)
const CLS_IDX = 46;        // индекс [CLS]
const PAD_IDX = 45;        // индекс [PAD]
const UNK_IDX = 44;        // индекс [UNK]
const MAX_SEQ_LEN = 64;    // из posEncoderPe
const D_MODEL = 48;
const D_FF = 192;

// ------------------------------------------------------------
// Токенизация с [CLS] и паддингом
// ------------------------------------------------------------
export const tokenize = (text: string): number[] => {
    // Построим обратный словарь символ -> индекс
    const charToIdx: { [key: string]: number } = {};
    for (const [idx, char] of Object.entries(vocab)) {
        charToIdx[char] = parseInt(idx);
    }

    const tokens: number[] = [CLS_IDX]; // начинаем с [CLS]
    for (const char of text) {
        tokens.push(charToIdx[char] ?? UNK_IDX);
    }

    // Обрезаем, если превышает MAX_SEQ_LEN
    if (tokens.length > MAX_SEQ_LEN) {
        return tokens.slice(0, MAX_SEQ_LEN);
    }

    // Дополняем паддингами до MAX_SEQ_LEN
    while (tokens.length < MAX_SEQ_LEN) {
        tokens.push(PAD_IDX);
    }
    return tokens;
};

// ------------------------------------------------------------
// Преобразование индексов в эмбеддинги (с масштабированием)
// ------------------------------------------------------------
export const convertToEmb = (tokenIndices: number[]): number[][] => {
    const embArr: number[][] = [];
    const scale = Math.sqrt(D_MODEL);
    for (let i = 0; i < tokenIndices.length; i++) {
        const idx = tokenIndices[i];
        const emb = embeddingWeight[idx].map(v => v * scale);
        embArr.push(emb);
    }
    return embArr;
};

// ------------------------------------------------------------
// Добавление позиционного кодирования
// ------------------------------------------------------------
export const addPositionalEncoding = (embs: number[][]): number[][] => {
    const result: number[][] = [];
    for (let i = 0; i < embs.length; i++) {
        const row = embs[i].map((val, j) => val + posEncoderPe[i][j]);
        result.push(row);
    }
    return result;
};

// ------------------------------------------------------------
// Вспомогательные функции
// ------------------------------------------------------------
const transpose = (matrix: number[][]): number[][] => {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result: number[][] = Array.from({ length: cols }, () => Array(rows));
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
};

const matMul = (a: number[][], b: number[][]): number[][] => {
    const rowsA = a.length;
    const colsA = a[0].length;
    const rowsB = b.length;
    const colsB = b[0].length;
    if (colsA !== rowsB) throw new Error("Несовместимые размеры для умножения матриц");
    const result: number[][] = Array.from({ length: rowsA }, () => Array(colsB).fill(0));
    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            let sum = 0;
            for (let k = 0; k < colsA; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
};

// Универсальное сложение: поддерживает матрицу + матрицу и матрицу + вектор (bias)
const matAdd = (a: number[][], b: number[] | number[][]): number[][] => {
    if (Array.isArray(b) && typeof b[0] === 'number') {
        // b - одномерный массив (bias)
        const bias = b as number[];
        return a.map(row => row.map((val, j) => val + bias[j]));
    } else {
        // b - двумерная матрица
        const bMat = b as number[][];
        return a.map((row, i) => row.map((val, j) => val + bMat[i][j]));
    }
};

const gelu = (x: number): number => {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
};

const softmax = (matrix: number[][]): number[][] => {
    return matrix.map(row => {
        const maxVal = Math.max(...row);
        const expRow = row.map(v => Math.exp(v - maxVal));
        const sumExp = expRow.reduce((s, v) => s + v, 0);
        return expRow.map(v => v / sumExp);
    });
};

const layerNorm = (matrix: number[][], gamma: number[], beta: number[]): number[][] => {
    const eps = 1e-8;
    return matrix.map(row => {
        const mean = row.reduce((s, v) => s + v, 0) / row.length;
        const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length;
        const std = Math.sqrt(variance + eps);
        return row.map((v, j) => ((v - mean) / std) * gamma[j] + beta[j]);
    });
};

// ------------------------------------------------------------
// Применение одного слоя трансформера (post-norm)
// ------------------------------------------------------------
const applyTransformerLayer = (
    x: number[][],                       // [seq_len, d_model]
    padMask: boolean[],                   // true для паддинг-токенов
    // Веса внимания
    Wqkv: number[][],                     // [3*d_model, d_model]
    bqkv: number[],                       // [3*d_model]
    Wo: number[][],                       // [d_model, d_model]
    bo: number[],                         // [d_model]
    // LayerNorm после attention
    gamma1: number[],                     // [d_model]
    beta1: number[],                      // [d_model]
    // LayerNorm после FFN
    gamma2: number[],                     // [d_model]
    beta2: number[],                      // [d_model]
    // FFN первый слой
    W1: number[][],                       // [d_ff, d_model]
    b1: number[],                         // [d_ff]
    // FFN второй слой
    W2: number[][],                       // [d_model, d_ff]
    b2: number[]                          // [d_model]
): number[][] => {
    const seqLen = x.length;
    const dModel = x[0].length;

    // ---- Внимание ----
    // Разделяем Wqkv на Q, K, V (каждая размерности [d_model, d_model])
    const Wq = Wqkv.slice(0, dModel);
    const Wk = Wqkv.slice(dModel, 2 * dModel);
    const Wv = Wqkv.slice(2 * dModel, 3 * dModel);
    const bq = bqkv.slice(0, dModel);
    const bk = bqkv.slice(dModel, 2 * dModel);
    const bv = bqkv.slice(2 * dModel, 3 * dModel);

    // Линейные проекции с bias (bias передаётся как одномерный массив)
    const Q = matAdd(matMul(x, transpose(Wq)), bq); // [seq_len, d_model]
    const K = matAdd(matMul(x, transpose(Wk)), bk);
    const V = matAdd(matMul(x, transpose(Wv)), bv);

    // Скоринг
    let scores = matMul(Q, transpose(K)); // [seq_len, seq_len]
    scores = scores.map(row => row.map(v => v / Math.sqrt(dModel)));

    // Маска паддинга: для i-го запроса, если j-й ключ – паддинг, зануляем
    for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
            if (padMask[j]) {
                scores[i][j] = -Infinity;
            }
        }
    }

    const attnWeights = softmax(scores);
    const context = matMul(attnWeights, V); // [seq_len, d_model]

    // Выход attention проекция
    const attnOut = matAdd(matMul(context, transpose(Wo)), bo); // [seq_len, d_model]

    // Residual и LayerNorm (post-norm)
    const xAfterAttn = matAdd(x, attnOut); // [seq_len, d_model]
    const xNorm1 = layerNorm(xAfterAttn, gamma1, beta1);

    // ---- Полносвязный слой ----
    // Первый линейный слой (d_model -> d_ff)
    let hidden = matAdd(matMul(xNorm1, transpose(W1)), b1); // [seq_len, d_ff]
    hidden = hidden.map(row => row.map(v => gelu(v)));

    // Второй линейный слой (d_ff -> d_model)
    const ffnOut = matAdd(matMul(hidden, transpose(W2)), b2); // [seq_len, d_model]

    // Residual и LayerNorm
    const xAfterFFN = matAdd(xNorm1, ffnOut);
    const xNorm2 = layerNorm(xAfterFFN, gamma2, beta2);

    return xNorm2;
};

// ------------------------------------------------------------
// Полный прямой проход текстового энкодера
// ------------------------------------------------------------
export const textEncoder = (text: string): number[] => {
    // Токенизация
    const tokenIndices = tokenize(text);

    // Эмбеддинги + позиционное кодирование
    const embs = convertToEmb(tokenIndices);
    const x = addPositionalEncoding(embs); // [MAX_SEQ_LEN, D_MODEL]

    // Маска паддинга (true для паддинг-токенов)
    const padMask = tokenIndices.map(idx => idx === PAD_IDX);

    // Слой 1
    const out1 = applyTransformerLayer(
        x, padMask,
        transformerLayers0Self_attnIn_proj_weight, transformerLayers0Self_attnIn_proj_bias,
        transformerLayers0Self_attnOut_projWeight, transformerLayers0Self_attnOut_projBias,
        transformerLayers0Norm1Weight, transformerLayers0Norm1Bias,
        transformerLayers0Norm2Weight, transformerLayers0Norm2Bias,
        transformerLayers0Linear1Weight, transformerLayers0Linear1Bias,
        transformerLayers0Linear2Weight, transformerLayers0Linear2Bias
    );

    // Слой 2
    const out2 = applyTransformerLayer(
        out1, padMask,
        transformerLayers1Self_attnIn_proj_weight, transformerLayers1Self_attnIn_proj_bias,
        transformerLayers1Self_attnOut_projWeight, transformerLayers1Self_attnOut_projBias,
        transformerLayers1Norm1Weight, transformerLayers1Norm1Bias,
        transformerLayers1Norm2Weight, transformerLayers1Norm2Bias,
        transformerLayers1Linear1Weight, transformerLayers1Linear1Bias,
        transformerLayers1Linear2Weight, transformerLayers1Linear2Bias
    );

    // Слой 3
    const out3 = applyTransformerLayer(
        out2, padMask,
        transformerLayers2Self_attnIn_proj_weight, transformerLayers2Self_attnIn_proj_bias,
        transformerLayers2Self_attnOut_projWeight, transformerLayers2Self_attnOut_projBias,
        transformerLayers2Norm1Weight, transformerLayers2Norm1Bias,
        transformerLayers2Norm2Weight, transformerLayers2Norm2Bias,
        transformerLayers2Linear1Weight, transformerLayers2Linear1Bias,
        transformerLayers2Linear2Weight, transformerLayers2Linear2Bias
    );

    // Финальная LayerNorm
    const finalOut = layerNorm(out3, normWeight, normBias);

    // Берём выход для [CLS] токена (индекс 0)
    return finalOut[0];
};