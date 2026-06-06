'use client';

import dynamic from 'next/dynamic';

// Disable SSR untuk AzureMap karena azure-maps-control
// mengakses `window` saat module load — tidak tersedia di server Next.js
const AzureMap = dynamic(
  () => import('../components/AzureMaps/index'),
  {
    ssr: false,
    loading: () => (
      <div style={{
        width: '100%',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#f1f3f4',
        fontFamily: "'Inter', sans-serif",
        flexDirection: 'column',
        gap: '12px',
      }}>
        <div style={{
          width: '36px', height: '36px',
          border: '3px solid #dadce0',
          borderTop: '3px solid #4285f4',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
        }} />
        <span style={{ color: '#5f6368', fontSize: '14px' }}>Memuat peta...</span>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    ),
  }
);

export default function Home() {
  return (
    <main style={{ width: '100%', height: '100vh', position: 'relative' }}>
      <AzureMap />
    </main>
  );
}