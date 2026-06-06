'use client';

import React, { useEffect, useRef, useState } from 'react';
import 'azure-maps-control/dist/atlas.min.css';

// Interface untuk data lokasi
interface LocationData {
  lon: number;
  lat: number;
}

interface PredictionResult {
  isSlum: boolean;
  confidence?: number;
  model_version?: string;
}

export default function SlumMap() {
  const mapRef = useRef<HTMLDivElement>(null);
  const [map, setMap] = useState<any>(null);
  const [dataSource, setDataSource] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Kita panggil library di dalam useEffect agar window pasti tersedia
    const atlas = require('azure-maps-control');

    if (!mapRef.current) return;

    // Inisialisasi Peta
    const azureMap = new atlas.Map(mapRef.current, {
      center: [106.8272, -6.1751], // Jakarta sebagai titik awal
      zoom: 12,
      style: 'road',
      authOptions: {
        authType: 'subscriptionKey' as any,
        subscriptionKey: process.env.NEXT_PUBLIC_AZURE_MAPS_KEY
      }
    });

    azureMap.events.add('ready', () => {
      // Setup Data Source untuk Marker/Pin
      const ds = new atlas.source.DataSource();
      azureMap.sources.add(ds);
      setDataSource(ds);

      // Layer visual untuk menampilkan Pin di atas peta
      azureMap.layers.add(new atlas.layer.SymbolLayer(ds, undefined, {
        iconOptions: {
          image: 'pin-blue',
          size: 1.2
        }
      }));

      // Tambahkan Kontrol Navigasi (Zoom, Pitch, Kompas)
      azureMap.controls.add([
        new atlas.control.ZoomControl(),
        new atlas.control.PitchControl(),
        new atlas.control.CompassControl()
      ], { position: 'top-right' as any });

      setMap(azureMap);
    });

    return () => {
      if (azureMap) azureMap.dispose();
    };
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery || !map || !dataSource) return;

    setIsLoading(true);
    setPrediction(null);
    const atlas = require('azure-maps-control');

    try {
      // 1. Cari Lokasi via Azure Search API (Hanya Indonesia)
      const searchUrl = `https://atlas.microsoft.com/search/address/json?api-version=1.0&subscription-key=${process.env.NEXT_PUBLIC_AZURE_MAPS_KEY}&query=${encodeURIComponent(searchQuery)}&countrySet=ID`;

      const response = await fetch(searchUrl);
      const data = await response.json();

      if (data.results && data.results.length > 0) {
        const { position } = data.results[0];
        const location: LocationData = { lon: position.lon, lat: position.lat };

        // 2. Animasi Terbang (Smooth Navigation)
        map.setCamera({
          center: [location.lon, location.lat],
          zoom: 15,
          type: 'fly',
          duration: 2000
        });

        // 3. Update Pin di Peta
        dataSource.clear();
        dataSource.add(new atlas.data.Point([location.lon, location.lat]));

        // 4. Panggil API Prediksi (Mocking untuk sekarang)
        await requestPrediction(location.lat, location.lon);
      } else {
        alert('Wilayah tidak ditemukan. Coba masukkan nama Kecamatan atau Kelurahan.');
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const requestPrediction = async (lat: number, lon: number) => {
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lon }),
      });
      const data = await res.json();
      setPrediction(data);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-zinc-100 font-sans">
      {/* UI Overlay: Search & Result */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 z-20 w-full max-w-lg px-4">
        <form
          onSubmit={handleSearch}
          className="flex bg-white/90 backdrop-blur-md rounded-2xl shadow-2xl p-2 border border-white/20 transition-all hover:shadow-zinc-200"
        >
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Cari Kecamatan atau Kelurahan..."
            className="flex-1 px-4 py-2 bg-transparent outline-none text-zinc-800 placeholder-zinc-400"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="bg-zinc-900 text-white px-6 py-2 rounded-xl font-medium transition-colors hover:bg-zinc-800 disabled:bg-zinc-300"
          >
            {isLoading ? 'Analisis...' : 'Cari'}
          </button>
        </form>

        {/* Floating Result Card */}
        {prediction && (
          <div className="mt-4 animate-in fade-in slide-in-from-top-4 duration-500">
            <div className={`p-5 rounded-2xl shadow-2xl border-l-[6px] backdrop-blur-lg ${prediction.isSlum
              ? 'bg-white/90 border-rose-500'
              : 'bg-white/90 border-emerald-500'
              }`}>
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-400 block mb-1">
                Kondisi Wilayah
              </span>
              <div className="flex items-center justify-between">
                <h2 className={`text-2xl font-black tracking-tight ${prediction.isSlum ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {prediction.isSlum ? 'ZONA KUMUH' : 'ZONA NON-KUMUH'}
                </h2>
                <div className="text-[10px] bg-zinc-100 px-2 py-1 rounded-md text-zinc-500 font-bold">
                  {prediction.model_version ? prediction.model_version.toUpperCase() : 'MODEL V1.0'}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Peta Azure Maps */}
      <div ref={mapRef} className="w-full h-full" />

      {/* Footer info tipis */}
      <div className="absolute bottom-4 left-4 z-10 hidden md:block">
        <p className="text-[10px] text-zinc-400 font-medium bg-white/50 backdrop-blur px-2 py-1 rounded-md">
          Azure Maps SDK • Wilayah Indonesia
        </p>
      </div>
    </div>
  );
}