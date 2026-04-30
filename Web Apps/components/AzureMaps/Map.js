'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as atlas from 'azure-maps-control';
import 'azure-maps-control/dist/atlas.min.css';
import {
  Search, X, MapPin, Loader2,
  AlertCircle, CheckCircle2, TriangleAlert,
  Navigation, Layers, Scan, ChevronDown, ChevronUp
} from 'lucide-react';
import { validateIndonesianLocation, hasAdministrativeKeyword } from '../../utils/locationValidator';

/* ═══════════════════════════════════════════════════════════════
   TOAST
═══════════════════════════════════════════════════════════════ */
const TOAST_META = {
  error:   { color: '#ef4444', bg: '#fff1f2', border: '#fecdd3', Icon: AlertCircle,   label: 'Error'     },
  success: { color: '#16a34a', bg: '#f0fdf4', border: '#bbf7d0', Icon: CheckCircle2,  label: 'Berhasil'  },
  warning: { color: '#d97706', bg: '#fffbeb', border: '#fde68a', Icon: TriangleAlert, label: 'Peringatan'},
};

function Toast({ toasts, onDismiss }) {
  if (!toasts.length) return null;
  return (
    <div style={{
      position:'fixed', bottom:'24px', right:'16px',
      zIndex:9999, display:'flex', flexDirection:'column', gap:'8px',
      pointerEvents:'none', maxWidth:'calc(100vw - 32px)',
    }}>
      {toasts.map((t) => {
        const m = TOAST_META[t.type] || TOAST_META.error;
        const { Icon } = m;
        return (
          <div key={t.id} style={{
            pointerEvents:'all', background:m.bg,
            border:`1px solid ${m.border}`, borderLeft:`4px solid ${m.color}`,
            borderRadius:'12px', boxShadow:'0 4px 20px rgba(0,0,0,0.10)',
            overflow:'hidden', animation:'toastIn 0.35s cubic-bezier(0.34,1.56,0.64,1) forwards',
            minWidth:'280px', maxWidth:'360px', fontFamily:"'Inter',sans-serif",
          }}>
            <div style={{display:'flex',alignItems:'flex-start',padding:'12px 14px',gap:'10px'}}>
              <Icon size={18} color={m.color} style={{flexShrink:0,marginTop:'1px'}} />
              <div style={{flex:1}}>
                <div style={{color:m.color,fontSize:'11px',fontWeight:'700',letterSpacing:'0.6px',textTransform:'uppercase',marginBottom:'2px'}}>{m.label}</div>
                <div style={{color:'#374151',fontSize:'13px',lineHeight:'1.45'}}>{t.message}</div>
              </div>
              <button onClick={()=>onDismiss(t.id)} style={{background:'none',border:'none',cursor:'pointer',color:'#9ca3af',padding:0,flexShrink:0,display:'flex',alignItems:'center'}}>
                <X size={15}/>
              </button>
            </div>
            <div style={{height:'3px',background:`${m.color}22`}}>
              <div style={{height:'100%',background:m.color,animation:`toastProg ${t.duration}ms linear forwards`,opacity:0.6}}/>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function useToast() {
  const [toasts, setToasts] = useState([]);
  const timers = useRef({});
  const dismiss = useCallback((id) => {
    clearTimeout(timers.current[id]);
    delete timers.current[id];
    setToasts(p => p.filter(t => t.id !== id));
  }, []);
  const showToast = useCallback((message, type='error', duration=4500) => {
    const id = Date.now().toString();
    setToasts(p => [...p, { id, message, type, duration }]);
    timers.current[id] = setTimeout(() => dismiss(id), duration);
  }, [dismiss]);
  return { toasts, showToast, dismiss };
}

/* ═══════════════════════════════════════════════════════════════
   SLUM ANALYSIS PANEL
═══════════════════════════════════════════════════════════════ */
function SlumPanel({ result, isAnalyzing, onAnalyze, onClear, currentLocation }) {
  const [expanded, setExpanded] = useState(true);

  const slumRatio = result?.metadata?.slum_ratio ?? null;
  const slumPct   = slumRatio !== null ? (slumRatio * 100).toFixed(1) : null;
  const featureCount = result?.features?.length ?? 0;

  const getRiskLevel = (pct) => {
    if (pct === null) return null;
    if (pct >= 30) return { label: 'Risiko Tinggi', color: '#ef4444', bg: '#fff1f2' };
    if (pct >= 10) return { label: 'Risiko Sedang', color: '#d97706', bg: '#fffbeb' };
    return { label: 'Risiko Rendah', color: '#16a34a', bg: '#f0fdf4' };
  };
  const risk = getRiskLevel(parseFloat(slumPct));

  return (
    <div style={{
      position:'absolute', bottom:'24px', left:'16px',
      width:'300px', background:'#fff', borderRadius:'16px',
      boxShadow:'0 4px 24px rgba(0,0,0,0.15)', fontFamily:"'Inter',sans-serif",
      zIndex:10, overflow:'hidden',
    }}>
      {/* Header */}
      <div style={{
        background:'linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%)',
        padding:'14px 16px', display:'flex', alignItems:'center', gap:'10px',
      }}>
        <div style={{background:'rgba(255,255,255,0.15)',borderRadius:'10px',padding:'7px',display:'flex'}}>
          <Scan size={18} color="#fff"/>
        </div>
        <div style={{flex:1}}>
          <div style={{color:'#fff',fontWeight:'700',fontSize:'14px'}}>Deteksi Kawasan Kumuh</div>
          <div style={{color:'rgba(255,255,255,0.6)',fontSize:'11px'}}>AI Satellite Analysis</div>
        </div>
        <button onClick={()=>setExpanded(e=>!e)} style={{background:'none',border:'none',cursor:'pointer',color:'rgba(255,255,255,0.7)',display:'flex'}}>
          {expanded ? <ChevronDown size={16}/> : <ChevronUp size={16}/>}
        </button>
      </div>

      {expanded && (
        <div style={{padding:'14px 16px'}}>
          {/* Analyze button */}
          <button
            onClick={onAnalyze}
            disabled={isAnalyzing || !currentLocation}
            style={{
              width:'100%', padding:'10px', border:'none', borderRadius:'10px',
              background: isAnalyzing ? '#e5e7eb' : 'linear-gradient(135deg,#667eea,#764ba2)',
              color: isAnalyzing ? '#9ca3af' : '#fff',
              fontWeight:'600', fontSize:'13px', cursor: isAnalyzing || !currentLocation ? 'not-allowed' : 'pointer',
              display:'flex', alignItems:'center', justifyContent:'center', gap:'8px',
              transition:'opacity 0.2s', fontFamily:"'Inter',sans-serif",
            }}
          >
            {isAnalyzing
              ? <><Loader2 size={15} style={{animation:'spin 0.8s linear infinite'}}/> Menganalisis...</>
              : <><Scan size={15}/> Analisis Area Ini</>
            }
          </button>

          {!currentLocation && (
            <p style={{fontSize:'11px',color:'#9ca3af',textAlign:'center',margin:'8px 0 0',lineHeight:'1.4'}}>
              Cari lokasi terlebih dahulu untuk mengaktifkan analisis
            </p>
          )}

          {/* Results */}
          {result && !isAnalyzing && (
            <div style={{marginTop:'12px'}}>
              <div style={{
                background: risk?.bg || '#f9fafb',
                border:`1px solid ${risk?.color || '#e5e7eb'}22`,
                borderRadius:'10px', padding:'10px 12px', marginBottom:'10px',
              }}>
                <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'4px'}}>
                  <span style={{fontSize:'11px',color:'#6b7280',fontWeight:'600',textTransform:'uppercase',letterSpacing:'0.5px'}}>Status</span>
                  <span style={{
                    background:risk?.color||'#6b7280', color:'#fff',
                    borderRadius:'20px', padding:'2px 8px', fontSize:'10px', fontWeight:'700',
                  }}>{risk?.label||'–'}</span>
                </div>
                <div style={{display:'flex',alignItems:'baseline',gap:'4px'}}>
                  <span style={{fontSize:'28px',fontWeight:'800',color:risk?.color||'#374151',lineHeight:1}}>
                    {slumPct ?? '–'}
                  </span>
                  <span style={{fontSize:'14px',color:'#6b7280',fontWeight:'500'}}>% kawasan kumuh</span>
                </div>
              </div>

              {/* Progress bar */}
              <div style={{marginBottom:'10px'}}>
                <div style={{display:'flex',justifyContent:'space-between',marginBottom:'4px'}}>
                  <span style={{fontSize:'11px',color:'#6b7280'}}>Proporsi area</span>
                  <span style={{fontSize:'11px',color:'#374151',fontWeight:'600'}}>{slumPct}%</span>
                </div>
                <div style={{height:'6px',background:'#f3f4f6',borderRadius:'99px',overflow:'hidden'}}>
                  <div style={{
                    height:'100%', width:`${Math.min(slumPct,100)}%`,
                    background:`linear-gradient(90deg,${risk?.color||'#6b7280'},${risk?.color||'#6b7280'}aa)`,
                    borderRadius:'99px', transition:'width 0.8s ease',
                  }}/>
                </div>
              </div>

              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'8px',marginBottom:'10px'}}>
                {[
                  {label:'Segmen Terdeteksi', val:featureCount.toLocaleString()},
                  {label:'Confidence', val:`${((result.metadata?.slum_confidence_mean||0)*100).toFixed(0)}%`},
                ].map(({label,val})=>(
                  <div key={label} style={{background:'#f9fafb',borderRadius:'8px',padding:'8px 10px'}}>
                    <div style={{fontSize:'10px',color:'#9ca3af',marginBottom:'2px'}}>{label}</div>
                    <div style={{fontSize:'15px',fontWeight:'700',color:'#374151'}}>{val}</div>
                  </div>
                ))}
              </div>

              <button onClick={onClear} style={{
                width:'100%',padding:'7px',border:'1px solid #e5e7eb',borderRadius:'8px',
                background:'#fff',color:'#6b7280',fontSize:'12px',cursor:'pointer',fontFamily:"'Inter',sans-serif",
              }}>
                Hapus Overlay
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN MAP COMPONENT
═══════════════════════════════════════════════════════════════ */
export default function Map() {
  const mapRef       = useRef(null);
  const mapInstance  = useRef(null);
  const datasource   = useRef(null);
  const slumSource   = useRef(null);
  const inputRef     = useRef(null);

  const [query,          setQuery]         = useState('');
  const [isSearching,    setSearching]     = useState(false);
  const [isFocused,      setFocused]       = useState(false);
  const [isMobile,       setMobile]        = useState(false);
  const [currentLocation, setCurrentLocation] = useState(null); // {lat, lon, name}
  const [isAnalyzing,    setAnalyzing]     = useState(false);
  const [slumResult,     setSlumResult]    = useState(null);

  const { toasts, showToast, dismiss } = useToast();
  const subscriptionKey = process.env.NEXT_PUBLIC_AZURE_MAPS_KEY;

  /* ── Responsive ─────────────────────────────────────────── */
  useEffect(() => {
    const check = () => setMobile(window.innerWidth < 640);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  /* ── Init Map ───────────────────────────────────────────── */
  useEffect(() => {
    if (mapInstance.current || !mapRef.current) return;

    const map = new atlas.Map(mapRef.current, {
      center: [118.0, -2.5], zoom: 5, view: 'Auto', style: 'road',
      authOptions: { authType: 'subscriptionKey', subscriptionKey },
    });
    mapInstance.current = map;

    map.events.add('ready', () => {
      map.controls.add(
        [new atlas.control.ZoomControl(), new atlas.control.CompassControl()],
        { position: 'bottom-right' }
      );

      // Search marker datasource
      const ds = new atlas.source.DataSource();
      map.sources.add(ds);
      datasource.current = ds;
      map.layers.add(new atlas.layer.SymbolLayer(ds, null, {
        iconOptions: { image: 'pin-red', anchor: 'bottom', allowOverlap: true },
      }));

      // Slum overlay datasource
      const slumDs = new atlas.source.DataSource();
      map.sources.add(slumDs);
      slumSource.current = slumDs;

      // Fill layer – semi-transparent red
      map.layers.add(new atlas.layer.PolygonLayer(slumDs, 'slum-fill', {
        fillColor: 'rgba(239,68,68,0.35)',
        fillOpacity: 0.7,
        filter: ['==', ['get', 'class'], 'slum'],
      }));
      // Outline
      map.layers.add(new atlas.layer.LineLayer(slumDs, 'slum-outline', {
        strokeColor: '#dc2626',
        strokeWidth: 1,
        filter: ['==', ['get', 'class'], 'slum'],
      }));
    });

    return () => { mapInstance.current?.dispose(); mapInstance.current = null; };
  }, []);

  /* ── Search ─────────────────────────────────────────────── */
  const handleSearch = async (e) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;

    const bypassed = hasAdministrativeKeyword(q);
    if (!bypassed) {
      const v = validateIndonesianLocation(q);
      if (!v.valid) {
        const hint = v.matched ? ` Maksud kamu "${v.matched}"?` : '';
        showToast(`"${q}" tidak dikenali sebagai lokasi di Indonesia.${hint}`, 'error');
        return;
      }
    }

    setSearching(true);
    inputRef.current?.blur();

    try {
      const url = `https://atlas.microsoft.com/search/fuzzy/json?api-version=1.0&query=${encodeURIComponent(q)}&subscription-key=${subscriptionKey}&language=id-ID&countrySet=ID&limit=1`;
      const res  = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.results?.length > 0) {
        const { lat, lon } = data.results[0].position;
        const name = data.results[0].address?.freeformAddress || q;
        mapInstance.current.setCamera({ center: [lon, lat], zoom: 15, type: 'fly', duration: 2000 });
        datasource.current?.clear();
        datasource.current?.add(new atlas.data.Feature(new atlas.data.Point([lon, lat])));
        setCurrentLocation({ lat, lon, name });
        setSlumResult(null);
        slumSource.current?.clear();
        showToast(`Menuju ${name}`, 'success');
      } else {
        showToast(`"${q}" tidak ditemukan. Coba nama yang lebih spesifik.`, 'error');
      }
    } catch (err) {
      showToast('Gagal menghubungi layanan pencarian. Periksa koneksimu.', 'warning');
    } finally {
      setSearching(false);
    }
  };

  /* ── Slum Analysis ─────────────────────────────────────── */
  const handleAnalyze = useCallback(async () => {
    if (!currentLocation || isAnalyzing) return;
    setAnalyzing(true);
    slumSource.current?.clear();

    try {
      const { lat, lon } = currentLocation;
      const res = await fetch(
        `/api/predict?lat=${lat}&lon=${lon}&radius_km=1.5`
      );
      const data = await res.json();

      if (data.error) throw new Error(data.error);

      // Add GeoJSON features to map
      if (data.features?.length > 0) {
        data.features.forEach(feat => {
          slumSource.current?.add(new atlas.data.Feature(
            new atlas.data.Polygon(feat.geometry.coordinates),
            feat.properties
          ));
        });
        setSlumResult(data);
        const pct = ((data.metadata?.slum_ratio || 0) * 100).toFixed(1);
        showToast(`Analisis selesai — ${pct}% area terdeteksi kumuh`, 'success');
      } else {
        setSlumResult(data);
        showToast('Tidak terdeteksi kawasan kumuh di area ini.', 'success');
      }
    } catch (err) {
      showToast(`Gagal menghubungi backend ML: ${err.message}`, 'error');
    } finally {
      setAnalyzing(false);
    }
  }, [currentLocation, isAnalyzing, showToast]);

  const handleClearSlum = useCallback(() => {
    slumSource.current?.clear();
    setSlumResult(null);
  }, []);

  /* ── Render ─────────────────────────────────────────────── */
  return (
    <div style={{ position:'relative', width:'100%', height:'100%', fontFamily:"'Inter',sans-serif" }}>

      {/* Map canvas */}
      <div ref={mapRef} style={{ position:'absolute', inset:0 }} />

      {/* Search panel */}
      <div style={{
        position:'absolute',
        top: isMobile ? '12px' : '16px',
        left: isMobile ? '12px' : '16px',
        right: isMobile ? '12px' : 'auto',
        zIndex:10,
        width: isMobile ? 'auto' : '380px',
        maxWidth: isMobile ? '100%' : '380px',
      }}>
        <div style={{
          background:'#fff',
          borderRadius: isFocused ? '12px 12px 0 0' : '12px',
          boxShadow: isFocused
            ? '0 2px 6px rgba(0,0,0,0.20),0 8px 24px rgba(0,0,0,0.08)'
            : '0 2px 6px rgba(0,0,0,0.18)',
          transition:'box-shadow 0.2s,border-radius 0.2s',
          overflow:'hidden',
        }}>
          <form onSubmit={handleSearch}>
            <div style={{
              display:'flex', alignItems:'center',
              padding:'0 12px 0 16px', height:'52px', gap:'10px',
              borderBottom: isFocused ? '1px solid #e5e7eb' : '1px solid transparent',
              transition:'border-color 0.2s',
            }}>
              <MapPin size={20} color="#4285f4" style={{flexShrink:0}}/>
              <input
                ref={inputRef}
                type="text"
                placeholder="Cari lokasi di Indonesia..."
                value={query}
                onChange={e=>setQuery(e.target.value)}
                onFocus={()=>setFocused(true)}
                onBlur={()=>setTimeout(()=>setFocused(false),150)}
                style={{
                  flex:1, border:'none', outline:'none',
                  fontSize:'15px', color:'#202124',
                  fontFamily:"'Inter',sans-serif", background:'transparent', minWidth:0,
                }}
              />
              {query && !isSearching && (
                <button type="button" onClick={()=>{setQuery('');inputRef.current?.focus();}} style={{
                  background:'none',border:'none',cursor:'pointer',color:'#5f6368',
                  display:'flex',alignItems:'center',padding:'4px',borderRadius:'50%',flexShrink:0,
                }}
                  onMouseEnter={e=>e.currentTarget.style.background='#f1f3f4'}
                  onMouseLeave={e=>e.currentTarget.style.background='none'}
                >
                  <X size={18}/>
                </button>
              )}
              {query && <div style={{width:'1px',height:'24px',background:'#dadce0',flexShrink:0}}/>}
              <button type="submit" disabled={isSearching||!query.trim()} style={{
                background:'none', border:'none',
                cursor:(!query.trim()||isSearching)?'not-allowed':'pointer',
                color:query.trim()?'#4285f4':'#9aa0a6',
                display:'flex',alignItems:'center',
                padding:'6px',borderRadius:'50%',flexShrink:0,
                transition:'background 0.15s,color 0.15s',
              }}
                onMouseEnter={e=>{if(query.trim())e.currentTarget.style.background='#e8f0fe';}}
                onMouseLeave={e=>e.currentTarget.style.background='none'}
              >
                {isSearching
                  ? <Loader2 size={20} style={{animation:'spin 0.8s linear infinite'}}/>
                  : <Search size={20}/>
                }
              </button>
            </div>
          </form>

          {isFocused && !query && (
            <div style={{padding:'10px 16px 12px'}}>
              <div style={{display:'flex',alignItems:'center',gap:'12px',padding:'8px 0',color:'#5f6368'}}>
                <Navigation size={16} color="#9aa0a6" style={{flexShrink:0}}/>
                <span style={{fontSize:'14px'}}>Contoh: Antapani Bandung, Kec. Cilincing</span>
              </div>
              <div style={{display:'flex',alignItems:'center',gap:'12px',padding:'8px 0',color:'#5f6368'}}>
                <Layers size={16} color="#9aa0a6" style={{flexShrink:0}}/>
                <span style={{fontSize:'14px'}}>Cari kota, kecamatan, atau kelurahan</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Slum Analysis Panel */}
      {!isMobile && (
        <SlumPanel
          result={slumResult}
          isAnalyzing={isAnalyzing}
          onAnalyze={handleAnalyze}
          onClear={handleClearSlum}
          currentLocation={currentLocation}
        />
      )}

      {/* Mobile: floating analyze button */}
      {isMobile && currentLocation && (
        <button
          onClick={handleAnalyze}
          disabled={isAnalyzing}
          style={{
            position:'absolute', bottom:'24px', left:'50%', transform:'translateX(-50%)',
            zIndex:10, background:'linear-gradient(135deg,#667eea,#764ba2)',
            color:'#fff', border:'none', borderRadius:'99px',
            padding:'12px 24px', fontWeight:'700', fontSize:'14px',
            boxShadow:'0 4px 16px rgba(102,126,234,0.5)',
            cursor: isAnalyzing ? 'not-allowed' : 'pointer',
            display:'flex', alignItems:'center', gap:'8px',
            fontFamily:"'Inter',sans-serif",
          }}
        >
          {isAnalyzing
            ? <><Loader2 size={16} style={{animation:'spin 0.8s linear infinite'}}/> Menganalisis...</>
            : <><Scan size={16}/> Deteksi Kawasan Kumuh</>
          }
        </button>
      )}

      <Toast toasts={toasts} onDismiss={dismiss}/>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes toastIn {
          from { opacity:0; transform:translateY(12px) scale(0.97); }
          to   { opacity:1; transform:translateY(0) scale(1); }
        }
        @keyframes toastProg { from { width:100%; } to { width:0%; } }
        input::placeholder { color:#9aa0a6 !important; }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}