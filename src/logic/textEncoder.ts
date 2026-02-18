import { vocab } from "../matrix/textEncoder/vocab/vocab";

import { embeddings } from "../matrix/textEncoder/embeddings/embeddings"; //number[vocab.length][48]
import { positionMatrix } from "../matrix/textEncoder/embeddings/positionMatrix"; //number[64][48]

import { Wk1 } from "../matrix/textEncoder/layer_1/Wk1"; //number[48][48]
import { Wq1 } from "../matrix/textEncoder/layer_1/Wq1"; //number[48][48]
import { Wv1 } from "../matrix/textEncoder/layer_1/Wv1"; //number[48][48]
import { Wo1 } from "../matrix/textEncoder/layer_1/Wo1"; //number[48][48]
import { gammaFirst1 } from "../matrix/textEncoder/layer_1/gammaFirst1";  //number[48] 
import { gammaSecond1 } from "../matrix/textEncoder/layer_1/gammaSecond1";  //number[48] 
import { betaFirst1 } from "../matrix/textEncoder/layer_1/betaFirst1";  //number[48] 
import { betaSecond1 } from "../matrix/textEncoder/layer_1/betaSecond1";  //number[48] 
import { FFNinput1 } from "../matrix/textEncoder/layer_1/FFNinput1";  //number[48][192]
import { FFNoutput1 } from "../matrix/textEncoder/layer_1/FFNoutput1";  //number[192][48]

import { Wk2 } from "../matrix/textEncoder/layer_2/Wk2"; //number[48][48]
import { Wq2 } from "../matrix/textEncoder/layer_2/Wq2"; //number[48][48]
import { Wv2 } from "../matrix/textEncoder/layer_2/Wv2"; //number[48][48]
import { Wo2 } from "../matrix/textEncoder/layer_2/Wo2"; //number[48][48]
import { gammaFirst2 } from "../matrix/textEncoder/layer_2/gammaFirst2";  //number[48] 
import { gammaSecond2 } from "../matrix/textEncoder/layer_2/gammaSecond2";  //number[48] 
import { betaFirst2 } from "../matrix/textEncoder/layer_2/betaFirst2";  //number[48] 
import { betaSecond2 } from "../matrix/textEncoder/layer_2/betaSecond2";  //number[48] 
import { FFNinput2 } from "../matrix/textEncoder/layer_2/FFNinput2";  //number[48][192]
import { FFNoutput2 } from "../matrix/textEncoder/layer_2/FFNoutput2";  //number[192][48]

import { Wk3 } from "../matrix/textEncoder/layer_3/Wk3"; //number[48][48]
import { Wq3 } from "../matrix/textEncoder/layer_3/Wq3"; //number[48][48]
import { Wv3 } from "../matrix/textEncoder/layer_3/Wv3"; //number[48][48]
import { Wo3 } from "../matrix/textEncoder/layer_3/Wo3"; //number[48][48]
import { gammaFirst3 } from "../matrix/textEncoder/layer_3/gammaFirst3";  //number[48] 
import { gammaSecond3 } from "../matrix/textEncoder/layer_3/gammaSecond3";  //number[48] 
import { betaFirst3 } from "../matrix/textEncoder/layer_3/betaFirst3";  //number[48] 
import { betaSecond3 } from "../matrix/textEncoder/layer_3/betaSecond3";  //number[48] 
import { FFNinput3 } from "../matrix/textEncoder/layer_3/FFNinput3";  //number[48][192]
import { FFNoutput3 } from "../matrix/textEncoder/layer_3/FFNoutput3";  //number[192][48]


// токенизация текста
export const tokenize = (text: string): number[] => {
    const charToIdx: { [key: string]: number } = {};
    for (const [idx, char] of Object.entries(vocab)) {
        charToIdx[char] = parseInt(idx);
    }
    const tokens: number[] = [];
    for (const char of text) {
        tokens.push(charToIdx[char] ?? charToIdx['[UNK]']);
    }
    return tokens;
};

// конвертация индексов токена в эмбеддинги
export const convertToEmb = (text: string) => {
    const tokensIndex = tokenize(text)

    const embArr: number[][] = []
    for (let i = 0; i < tokensIndex.length; i++) {
        embArr.push(embeddings[tokensIndex[i]])
    }

    for (let i = 0; i < (positionMatrix.length - tokensIndex.length); i++) {
        embArr.push(new Array(48).fill(0))
    }

    return embArr
}

// складываем эмбеддинги и матрицу позиций
export const summWithPos = (text: string) => {
    const embs = convertToEmb(text)
    for (let i = 0; i < embs.length; i++) {
        for (let j = 0; j < embs[i].length; j++) {
            embs[i][j] += positionMatrix[i][j]
        }
    }

    return embs
}

// транспонирование матрицы
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

// умножение матриц
const matMul = (a: number[][], b: number[][]): number[][] => {
    const rowsA = a.length;
    const colsA = a[0].length;
    const rowsB = b.length;
    const colsB = b[0].length;
    if (colsA !== rowsB) throw new Error("Несовместимые размеры");
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

// сложение матриц
const matAdd = (a: number[][], b: number[][]): number[][] => {
    return a.map((row, i) => row.map((val, j) => val + b[i][j]));
};

const gelu = (x: number): number => {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
};

// softmax для матрицы
const softmax = (matrix: number[][]): number[][] => {
    return matrix.map(row => {
        const maxVal = Math.max(...row);
        const expRow = row.map(v => Math.exp(v - maxVal));
        const sumExp = expRow.reduce((s, v) => s + v, 0);
        return expRow.map(v => v / sumExp);
    });
};

// layer Norm
const layerNorm = (matrix: number[][], gamma: number[], beta: number[]): number[][] => {
    const eps = 1e-8;
    return matrix.map(row => {
        const mean = row.reduce((s, v) => s + v, 0) / row.length;
        const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length;
        const std = Math.sqrt(variance + eps);
        return row.map((v, j) => ((v - mean) / std) * gamma[j] + beta[j]);
    });
};

// применение одного слоя трансформера
const applyLayer = (x: number[][], real_len: number,
    Wq: number[][], Wk: number[][], Wv: number[][], Wo: number[][],
    gammaFirst: number[], betaFirst: number[],
    gammaSecond: number[], betaSecond: number[],
    FFNinput: number[][], FFNoutput: number[][]) => {
    // блок внимания
    const x_norm = layerNorm(x, gammaFirst, betaFirst);
    const seq_len = x.length;
    const d = x_norm[0].length;

    const Q = matMul(x_norm, Wq);
    const K = matMul(x_norm, Wk);
    const V = matMul(x_norm, Wv);

    // скоринг
    let scores = matMul(Q, transpose(K));
    scores = scores.map(row => row.map(v => v / Math.sqrt(d)));

    // маска внимания
    for (let i = 0; i < seq_len; i++) {
        for (let j = 0; j < seq_len; j++) {
            if (j >= real_len) {
                scores[i][j] = -Infinity;
            }
        }
    }

    const attn_weights = softmax(scores);
    const context = matMul(attn_weights, V);
    const attn_out = matMul(context, Wo);
    const x1 = matAdd(x, attn_out);

    // полносвязный слой
    // нормирование матриц
    const x1_norm = layerNorm(x1, gammaSecond, betaSecond);

    // первый линейный слой: 48 -> 192
    const W1 = FFNinput;
    let hidden = matMul(x1_norm, W1);
    hidden = hidden.map(row => row.map(v => gelu(v)));

    // второй линейный слой: 192 -> 48
    const W2 = FFNoutput;
    const ffn_out = matMul(hidden, W2);
    const x2 = matAdd(x1, ffn_out);

    return x2;
};

// полный прямой проход
export const textEncoder = (text: string): number[] => {
    const X = summWithPos(text);
    const real_len = tokenize(text).length;

    const outLayer1 = applyLayer(X, real_len, Wq1, Wk1, Wv1, Wo1, gammaFirst1, betaFirst1, gammaSecond1, betaSecond1, FFNinput1, FFNoutput1);
    const outLayer2 = applyLayer(outLayer1, real_len, Wq2, Wk2, Wv2, Wo2, gammaFirst2, betaFirst2, gammaSecond2, betaSecond2, FFNinput2, FFNoutput2);
    const outLayer3 = applyLayer(outLayer2, real_len, Wq3, Wk3, Wv3, Wo3, gammaFirst3, betaFirst3, gammaSecond3, betaSecond3, FFNinput3, FFNoutput3);

    return outLayer3[real_len - 1];
};
