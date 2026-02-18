import arr from "../data/arr.json"
// Берем весь массив датасета и возвращаем случайный индекс каждой цифры от 0 до 9

interface MnistSample {
    key: string;
    value: number[];
}
export const createRandomDigitsSample = () => {
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