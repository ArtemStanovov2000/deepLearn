import arr from "../data/arr.json"
import { textEncoder } from "./textEncoder"
import { imageEncoder } from "./imageEncoder"

const textDescriptionOfDigits = ["число ноль", "число один","число два","число три","число четыре","число пять","число шесть","число семь","число восемь","число девять"]

// Берем весь массив датасета и возвращаем случайный индекс каждой цифры от 0 до 9
const createRandomDigitsSample = () => {
    // Собираем индексы для каждой цифры
    const digitIndices: Record<string, number[]> = {};

    for (let i = 0; i < arr.length; i++) {
        const digit = arr[i].key;
        if (!digitIndices[digit]) {
            digitIndices[digit] = [];
        }
        digitIndices[digit].push(i);
    }

    const result: number[] = [];

    for (let d = 0; d <= 9; d++) {
        const digitStr = d.toString();
        const indices = digitIndices[digitStr];
        if (indices && indices.length > 0) {
            const randomIndex = Math.floor(Math.random() * indices.length);
            result.push(indices[randomIndex]);
        } else {
            console.warn(`Цифра ${d} отсутствует в массиве.`);
            result.push(-1); // маркер отсутствия
        }
    }

    return result; // массив из 10 индексов
}

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

export const calculateLoss = () => {
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

    similarityMatrix

    return similarityMatrix; // общая ошибка косинусного сходства матрицы 10х10 - примерно 2.543
}