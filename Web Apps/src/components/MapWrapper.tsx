'use client';

import dynamic from 'next/dynamic';

const DynamicMap = dynamic(() => import('./SlumMap'), { 
  ssr: false,
  loading: () => (
    <div className="h-screen w-full flex items-center justify-center bg-zinc-50">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-zinc-900 mx-auto mb-4"></div>
        <p className="text-zinc-500 font-medium">Menginisialisasi Peta Spasial...</p>
      </div>
    </div>
  )
});

export default function MapWrapper() {
  return <DynamicMap />;
}