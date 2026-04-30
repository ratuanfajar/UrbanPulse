/**
 * Levenshtein Distance menggunakan Dynamic Programming.
 *
 * dp[i][j] = minimum edit distance antara a[0..i-1] dan b[0..j-1]
 *
 * Recurrence:
 *   dp[i][0] = i  (hapus semua karakter a)
 *   dp[0][j] = j  (insert semua karakter b)
 *   dp[i][j] = dp[i-1][j-1]                         jika a[i-1] == b[j-1]
 *   dp[i][j] = 1 + min(dp[i-1][j],                  (delete)
 *                      dp[i][j-1],                   (insert)
 *                      dp[i-1][j-1])                 (replace)
 *
 * Kompleksitas: O(m × n) waktu, O(m × n) ruang
 */
export function levenshteinDistance(a, b) {
  const m = a.length;
  const n = b.length;

  // Buat tabel DP (m+1) × (n+1)
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => {
      if (i === 0) return j;
      if (j === 0) return i;
      return 0;
    })
  );

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] =
          1 +
          Math.min(
            dp[i - 1][j],     // delete dari a
            dp[i][j - 1],     // insert ke a
            dp[i - 1][j - 1]  // replace
          );
      }
    }
  }

  return dp[m][n];
}

/**
 * Similarity score antara dua string: rentang [0.0, 1.0]
 * 1.0 = identik, 0.0 = sama sekali berbeda
 */
export function stringSimilarity(a, b) {
  const longer = Math.max(a.length, b.length);
  if (longer === 0) return 1.0;
  const dist = levenshteinDistance(a, b);
  return 1 - dist / longer;
}

/* ─────────────────────────────────────────────────────────────
   Database Lokasi Indonesia
   Mencakup: 38 provinsi + ibu kota + kota besar + wilayah terkenal
───────────────────────────────────────────────────────────────*/
export const INDONESIA_LOCATIONS = [
  // ── PROVINSI ──────────────────────────────────────────────
  'aceh', 'sumatera utara', 'sumatera barat', 'riau',
  'kepulauan riau', 'jambi', 'sumatera selatan',
  'kepulauan bangka belitung', 'bengkulu', 'lampung',
  'dki jakarta', 'jakarta', 'jawa barat', 'banten',
  'jawa tengah', 'di yogyakarta', 'yogyakarta',
  'jawa timur', 'bali', 'nusa tenggara barat',
  'nusa tenggara timur', 'kalimantan barat',
  'kalimantan tengah', 'kalimantan selatan',
  'kalimantan timur', 'kalimantan utara',
  'sulawesi utara', 'sulawesi tengah', 'sulawesi selatan',
  'sulawesi tenggara', 'gorontalo', 'sulawesi barat',
  'maluku', 'maluku utara', 'papua', 'papua barat',
  'papua selatan', 'papua tengah', 'papua pegunungan',
  'papua barat daya',

  // ── SINGKATAN & ALIAS ─────────────────────────────────────
  'ntb', 'ntt', 'diy', 'kalbar', 'kalteng', 'kalsel',
  'kaltim', 'kaltara', 'sulut', 'sulteng', 'sulsel',
  'sultra', 'sulbar', 'sumut', 'sumbar', 'sumsel',
  'babel', 'kepri', 'jabar', 'jateng', 'jatim',

  // ── IBU KOTA PROVINSI ─────────────────────────────────────
  'banda aceh', 'medan', 'padang', 'pekanbaru',
  'tanjungpinang', 'palembang', 'pangkalpinang',
  'bandar lampung', 'serang', 'semarang', 'surabaya',
  'denpasar', 'mataram', 'kupang', 'pontianak',
  'palangkaraya', 'banjarmasin', 'samarinda',
  'tanjung selor', 'manado', 'palu', 'makassar',
  'kendari', 'mamuju', 'ambon', 'sofifi',
  'jayapura', 'manokwari', 'merauke', 'nabire',
  'wamena', 'sorong',

  // ── KOTA BESAR ────────────────────────────────────────────
  'bogor', 'depok', 'tangerang', 'tangerang selatan',
  'bekasi', 'bandung', 'cimahi', 'tasikmalaya',
  'cirebon', 'sukabumi', 'solo', 'surakarta', 'malang',
  'blitar', 'kediri', 'madiun', 'mojokerto', 'pasuruan',
  'probolinggo', 'batu', 'gresik', 'sidoarjo',
  'jember', 'banyuwangi', 'magelang', 'salatiga',
  'pekalongan', 'tegal', 'purwokerto', 'cilacap',
  'balikpapan', 'bontang', 'tarakan', 'batam',
  'bintan', 'dumai', 'pematangsiantar', 'tebing tinggi',
  'binjai', 'langsa', 'lhokseumawe', 'sabang',
  'sibolga', 'padangsidimpuan', 'gunungsitoli',
  'payakumbuh', 'bukittinggi', 'solok', 'sawah lunto',
  'pariaman', 'lubuklinggau', 'prabumulih', 'pagaralam',

  // ── WISATA & WILAYAH TERKENAL ─────────────────────────────
  'ubud', 'kuta', 'seminyak', 'nusa dua', 'jimbaran',
  'sanur', 'canggu', 'lembongan', 'uluwatu',
  'senggigi', 'lombok', 'gili trawangan', 'gili air',
  'labuan bajo', 'flores', 'komodo', 'raja ampat',
  'bunaken', 'likupang', 'toraja', 'tana toraja',
  'belitung', 'tanjung pandan', 'wakatobi',
  'ternate', 'tidore', 'morotai', 'halmahera',
  'pulau weh', 'pulau nias', 'nias', 'danau toba',
  'samosir', 'parapat', 'bromo', 'semeru', 'ijen',
  'karimunjawa', 'dieng', 'wonosobo', 'pangandaran',
  'garut', 'ciamis', 'kepulauan seribu',

  // ── KAWASAN POPULER ───────────────────────────────────────
  'glodok', 'kemang', 'menteng', 'kebayoran',
  'sudirman', 'thamrin', 'kuningan', 'mampang',
  'pasar minggu', 'cilandak', 'pondok indah',
  'kelapa gading', 'sunter', 'pluit', 'ancol',
  'kota tua jakarta', 'monas',
  'dago', 'braga', 'buah batu', 'cicaheum',
  'malioboro', 'prawirotaman',
  'tunjungan', 'rungkut', 'wonokromo',
];

/**
 * Validasi apakah query pengguna cukup mirip dengan
 * salah satu lokasi di Indonesia menggunakan DP Levenshtein.
 *
 * Strategi:
 * 1. Substring match langsung (instan, tanpa DP)
 * 2. Per-kata similarity dengan DP — ambil skor tertinggi
 * 3. Full-phrase similarity dengan DP
 *
 * @param {string} query  - input pengguna
 * @param {number} threshold - minimum similarity [0.0–1.0], default 0.55
 * @returns {{ valid: boolean, matched: string|null, score: number }}
 */
export function validateIndonesianLocation(query, threshold = 0.55) {
  // Normalisasi: lowercase, trim, collapse multiple spaces
  const q = query.toLowerCase().trim().replace(/\s+/g, ' ');
  if (!q || q.length < 2) return { valid: false, matched: null, score: 0 };

  const queryWords = q.split(' ').filter((w) => w.length >= 2);
  const isMultiWord = queryWords.length > 1;

  let bestScore = 0;
  let bestMatch = null;

  for (const loc of INDONESIA_LOCATIONS) {
    // ── 1. Substring check (O(n), tanpa DP) ───────────────────
    // "antapani bandung".includes("bandung") → true → langsung valid
    if (loc.includes(q) || q.includes(loc)) {
      return { valid: true, matched: loc, score: 1.0 };
    }

    const locWords = loc.split(' ');

    // ── 2. Per-kata DP similarity ──────────────────────────────
    for (const qw of queryWords) {
      for (const lw of locWords) {
        const score = stringSimilarity(qw, lw);
        if (score > bestScore) {
          bestScore = score;
          bestMatch = loc;
        }

        // Fast-pass untuk query multi-kata:
        // Jika SATU KATA saja sudah sangat mirip dengan lokasi Indonesia
        // (score ≥ 0.8), maka keseluruhan query dianggap valid.
        // Contoh: "antapani bandung" → "bandung" score 1.0 → pass
        //         "andir bandung"   → "bandung" score 1.0 → pass
        //         "padasuka cimahi" → "cimahi"  score 1.0 → pass
        if (isMultiWord && score >= 0.8) {
          return { valid: true, matched: loc, score };
        }
      }
    }

    // ── 3. Full-phrase similarity ──────────────────────────────
    const fullScore = stringSimilarity(q, loc);
    if (fullScore > bestScore) {
      bestScore = fullScore;
      bestMatch = loc;
    }
  }

  // Threshold adaptif:
  // - Multi-kata: lebih permisif (0.45) karena kecamatan+kota sudah cukup
  // - Single-kata: tetap ketat (threshold, default 0.55)
  const effectiveThreshold = isMultiWord ? 0.45 : threshold;

  return {
    valid: bestScore >= effectiveThreshold,
    matched: bestScore >= effectiveThreshold ? bestMatch : null,
    score: bestScore,
  };
}


/* ─────────────────────────────────────────────────────────────
   Kata kunci administratif → bypass validasi DP lokal
   Karena kecamatan/kelurahan/jalan terlalu banyak untuk di-hardcode,
   cukup deteksi prefix-nya lalu serahkan ke API (countrySet=ID).
───────────────────────────────────────────────────────────────*/
export const ADMINISTRATIVE_KEYWORDS = [
  // Level administratif Indonesia
  'kecamatan', 'kec',
  'kelurahan', 'kel',
  'desa', 'ds',
  'kabupaten', 'kab',
  'kotamadya', 'provinsi', 'prov',
  'rw', 'rt',

  // Jalan & alamat
  'jalan', 'jl',
  'gang', 'gg',
  'komplek', 'komp',
  'perumahan', 'perum',
  'blok', 'kavling', 'kav',

  // Fasilitas umum
  'pasar', 'terminal', 'stasiun', 'bandara', 'pelabuhan',
  'rumah sakit', 'puskesmas',
  'rs', 'rsud', 'rsu', 'rsup', 'rsia',
  'sekolah', 'sd', 'smp', 'sma', 'smk',
  'universitas', 'univ', 'institut', 'politeknik', 'akademi',
  'masjid', 'gereja', 'pura', 'vihara', 'klenteng',
  'hotel', 'mall', 'plaza', 'gedung', 'kantor',

  // Fitur geografis
  'taman', 'pantai', 'gunung', 'danau', 'sungai',
  'pulau', 'teluk', 'selat', 'bukit', 'lembah', 'hutan',
];

/**
 * Cek apakah query mengandung kata kunci administratif/POI
 * sehingga validasi DP lokal di-bypass → langsung ke Azure Maps API.
 *
 * Contoh yang di-bypass:
 *   "Kecamatan Cibinong"    → mengandung "kecamatan"
 *   "Jl. Sudirman No. 5"    → mengandung "jl"
 *   "RSUD Dr. Soetomo"      → mengandung "rsud"
 *   "Universitas Indonesia" → mengandung "universitas"
 *   "Gunung Rinjani"        → mengandung "gunung"
 *
 * @param {string} query
 * @returns {boolean}
 */
export function hasAdministrativeKeyword(query) {
  const q = query.toLowerCase().trim();
  // Tokenize: pisah berdasarkan spasi, koma, titik, garis bawah
  const words = q.split(/[\s,._]+/);
  return ADMINISTRATIVE_KEYWORDS.some((keyword) =>
    words.some((word) => word === keyword)
  );
}
