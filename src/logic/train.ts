import arr from "../data/arr.json"
import { textEncoder } from "./textEncoder"
import { imageEncoder } from "./imageEncoder"
import { createRandomDigitsSample } from "./createRandomDigitsSample"

const textDescriptionOfDigits = ["число ноль", "число один","число два","число три","число четыре","число пять","число шесть","число семь","число восемь","число девять"]

// Скалярное произведение двух векторов
function dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

// Евклидова норма вектора
function magnitude(vec: number[]): number {
    return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
}

// Косинусное сходство
function cosineSimilarity(a: number[], b: number[]): number {
    const dot = dotProduct(a, b);
    const magA = magnitude(a);
    const magB = magnitude(b);
    if (magA === 0 || magB === 0) return 0; // защита от деления на ноль
    return dot / (magA * magB);
}

// Вычисление общей ошибки
function contrastiveLoss(similarityMatrix: number[][], temperature: number = 0.1): number {
    const n = similarityMatrix.length; // ожидается 10
    let totalLoss = 0;

    // Text → Image (по строкам)
    for (let i = 0; i < n; i++) {
        const logits = similarityMatrix[i].map(s => s / temperature);
        const maxLogit = Math.max(...logits);                // для численной стабильности
        const expLogits = logits.map(l => Math.exp(l - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const prob = expLogits[i] / sumExp;                  // вероятность правильной пары
        totalLoss += -Math.log(prob);
    }

    // Image → Text (по столбцам)
    for (let j = 0; j < n; j++) {
        const logits = similarityMatrix.map(row => row[j] / temperature);
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(l => Math.exp(l - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const prob = expLogits[j] / sumExp;
        totalLoss += -Math.log(prob);
    }

    // Усредняем по 2n элементам
    return totalLoss / (2 * n);
}

export const train = () => {
    const textVectors = [];
    for (let i = 0; i < textDescriptionOfDigits.length; i++) {
        textVectors.push(textEncoder(textDescriptionOfDigits[i]));
    }

    const randomSample = createRandomDigitsSample();
    const imageVectors = [];
    for (let i = 0; i < randomSample.length; i++) {
        imageVectors.push(imageEncoder(arr[randomSample[i]].value));
    }

    // Матрица сходства: строки — тексты, столбцы — изображения
    const similarityMatrix: number[][] = [];
    for (let i = 0; i < textVectors.length; i++) {
        const row: number[] = [];
        for (let j = 0; j < imageVectors.length; j++) {
            row.push(cosineSimilarity(textVectors[i], imageVectors[j]));
        }
        similarityMatrix.push(row);
    }

    const loss = contrastiveLoss(similarityMatrix)

    return loss;
}