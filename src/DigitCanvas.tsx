import { useEffect, useRef, type FC } from 'react';

type Props = {
    pixels: number[],
    label: string
}

const DigitCanvas: FC<Props> = ({ pixels, label }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas: HTMLCanvasElement | null = canvasRef.current;
        if (canvas !== null) {
            const ctx: CanvasRenderingContext2D = canvas.getContext('2d');
            const imageData = ctx?.createImageData(28, 28);

            if (ctx && imageData) {
                // Заполняем ImageData значениями пикселей (grayscale)
                for (let i = 0; i < pixels.length; i++) {
                    const val = pixels[i];
                    imageData.data[i * 4] = val;       // R
                    imageData.data[i * 4 + 1] = val;   // G
                    imageData.data[i * 4 + 2] = val;   // B
                    imageData.data[i * 4 + 3] = 255;   // A
                }

                ctx.putImageData(imageData, 0, 0);
            }
        }
    }, [pixels]);

    return (
        <div style={{ textAlign: 'center', margin: '8px' }}>
            <canvas
                ref={canvasRef}
                width={28}
                height={28}
                style={{
                    border: '1px solid #ccc',
                    width: 56,            // увеличиваем для наглядности
                    height: 56,
                    imageRendering: 'pixelated' // сохраняем чёткость пикселей
                }}
            />
            <div>{label}</div>
        </div>
    );
};

export default DigitCanvas;