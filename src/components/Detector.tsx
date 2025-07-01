import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

const MODEL_URL = "/web_model/model.json";

const LABELS = [
	"10 Naira",
	"20 Naira",
	"50 Naira",
	"100 Naira",
	"200 Naira",
	"500 Naira",
	"1000 Naira",
];

const Detector: React.FC = () => {
	const videoRef = useRef<HTMLVideoElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [model, setModel] = useState<tf.GraphModel | null>(null);

	useEffect(() => {
		const setup = async () => {
			await tf.setBackend("webgl");
			await tf.ready();
			const loadedModel = await tf.loadGraphModel(MODEL_URL);
			setModel(loadedModel);

			const stream = await navigator.mediaDevices.getUserMedia({
				video: { facingMode: "environment" },
			});
			if (videoRef.current) videoRef.current.srcObject = stream;
		};

		setup();
	}, []);

	const detect = async () => {
		if (!model || !videoRef.current || !canvasRef.current) return;

		const video = videoRef.current;

		const inputTensor = tf.tidy(
			() =>
				tf.browser
					.fromPixels(video)
					.resizeBilinear([320, 320])
					.toFloat()
					.sub(127.5)
					.div(127.5)
					.expandDims(0), // shape [1, 320, 320, 3]
		);

		const [boxes, classes, scores] = (await model.executeAsync(
			inputTensor,
		)) as tf.Tensor[];

		const boxesData = (await boxes.array()) as number[][][];
		const classesData = (await classes.array()) as number[][];
		const scoresData = (await scores.array()) as number[][];

		const ctx = canvasRef.current.getContext("2d");
		if (!ctx) return;

		ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

		boxesData[0].forEach((box, i) => {
			const score = scoresData[0][i];
			if (score > 0.5) {
				const [ymin, xmin, ymax, xmax] = box;
				const x = xmin * video.width;
				const y = ymin * video.height;
				const width = (xmax - xmin) * video.width;
				const height = (ymax - ymin) * video.height;

				ctx.strokeStyle = "lime";
				ctx.lineWidth = 2;
				ctx.strokeRect(x, y, width, height);

				const classIndex = Math.round(classesData[0][i]);
				const label = LABELS[classIndex] || `Class ${classIndex}`;
				ctx.fillStyle = "black";
				ctx.fillText(`${label} (${Math.round(score * 100)}%)`, x + 4, y + 14);
			}
		});

		tf.dispose([inputTensor, boxes, scores, classes]);
		requestAnimationFrame(detect);
	};

	return (
		<div style={{ position: "relative", width: 320, height: 320 }}>
			<video
				ref={videoRef}
				width={320}
				height={320}
				autoPlay
				playsInline
				muted
				onLoadedData={detect}
				style={{ position: "absolute", top: 0, left: 0 }}
			/>
			<canvas
				ref={canvasRef}
				width={320}
				height={320}
				style={{ position: "absolute", top: 0, left: 0 }}
			/>
		</div>
	);
};

export default Detector;
