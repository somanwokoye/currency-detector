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
		const tensor = tf.tidy(() =>
			tf.browser
				.fromPixels(video)
				.resizeBilinear([300, 300]) // Match training size
				.toFloat()
				.div(255.0)
				.expandDims(0),
		);

		const predictions = (await model.executeAsync(
			tensor,
		)) as tf.Tensor<tf.Rank>[];

		const [boxes, scores, classes, num] = predictions;
		const boxesData = boxes.arraySync() as number[][][];
		const scoresData = scores.arraySync() as number[][];
		const classesData = classes.arraySync() as number[][];

		const ctx = canvasRef.current.getContext("2d");
		if (!ctx) return;
		ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

		boxesData[0].forEach((box, i) => {
			if (scoresData[0][i] > 0.5) {
				const [ymin, xmin, ymax, xmax] = box;
				const x = xmin * video.width;
				const y = ymin * video.height;
				const width = (xmax - xmin) * video.width;
				const height = (ymax - ymin) * video.height;

				ctx.strokeStyle = "lime";
				ctx.lineWidth = 2;
				ctx.strokeRect(x, y, width, height);

				const label = LABELS[Math.round(classesData[0][i]) - 1] || "Unknown";
				ctx.fillStyle = "black";
				ctx.fillText(label, x + 4, y + 12);
			}
		});

		tf.dispose(predictions);
		tf.dispose(tensor);
		requestAnimationFrame(detect);
	};

	return (
		<div style={{ position: "relative", width: 300, height: 300 }}>
			<video
				ref={videoRef}
				width={300}
				height={300}
				autoPlay
				playsInline
				muted
				onLoadedData={detect}
				style={{ position: "absolute" }}
			/>
			<canvas
				ref={canvasRef}
				width={300}
				height={300}
				style={{ position: "absolute" }}
			/>
		</div>
	);
};


export default Detector;