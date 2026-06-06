/**
 * Next.js API Route – proxy ke FastAPI backend-ml
 * GET  /api/predict?lat=&lon=&radius_km=   → demo prediction
 * POST /api/predict                         → file upload prediction
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_ML_BACKEND_URL || 'http://127.0.0.1:8000';

// ── GET /api/predict (demo mode) ─────────────────────────────
export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const lat       = searchParams.get('lat')       || '-6.2';
  const lon       = searchParams.get('lon')       || '106.8';
  const radius_km = searchParams.get('radius_km') || '1.5';

  try {
    const res = await fetch(
      `${BACKEND_URL}/predict/demo?lat=${lat}&lon=${lon}&radius_km=${radius_km}`,
      { cache: 'no-store' }
    );
    if (!res.ok) {
      const text = await res.text();
      return Response.json({ error: `Backend error ${res.status}: ${text}` }, { status: res.status });
    }
    const data = await res.json();
    return Response.json(data);
  } catch (err) {
    return Response.json({ error: `Cannot reach ML backend: ${err.message}` }, { status: 503 });
  }
}

// ── POST /api/predict (file upload mode) ─────────────────────
export async function POST(request) {
  try {
    const formData = await request.formData();
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const text = await res.text();
      return Response.json({ error: `Backend error ${res.status}: ${text}` }, { status: res.status });
    }
    const data = await res.json();
    return Response.json(data);
  } catch (err) {
    return Response.json({ error: `Cannot reach ML backend: ${err.message}` }, { status: 503 });
  }
}
