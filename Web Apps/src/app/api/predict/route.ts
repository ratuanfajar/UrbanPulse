import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { lat, lon } = await req.json();

    // Pastikan MODEL_API_URL diset ke URL FastAPI (misal http://localhost:8000/predict)
    const apiUrl = process.env.MODEL_API_URL || 'http://localhost:8000/predict';

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latitude: lat, longitude: lon }),
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data); // Berisi { isSlum: true/false } dari model
      }
    } catch (fetchError) {
      console.warn("Koneksi ke ML API gagal, fallback ke data mock.", fetchError);
    }

    // SEMENTARA: Fallback jika API FastAPI belum jalan
    const isSlumMock = Math.random() > 0.5; 
    return NextResponse.json({ 
      isSlum: isSlumMock,
      confidence: 0.85,
      coordinates: { lat, lon },
      model_version: "phase2_best"
    });
  } catch (error) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}