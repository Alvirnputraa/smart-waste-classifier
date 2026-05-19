import { useState, useRef, useCallback, useEffect } from 'react'
import axios from 'axios'
import styles from './App.module.css'

// Gunakan IP lokal agar bisa diakses dari HP di jaringan yang sama
// Untuk akses kamera di Android, gunakan URL HTTPS dari ngrok
// Ganti dengan URL ngrok backend kamu jika pakai ngrok, contoh:
// const API_URL = 'https://xyz789.ngrok-free.app'
const API_URL = import.meta.env.VITE_API_URL || '/api'

// ─── Icons ────────────────────────────────────────────────────────────────────
const UploadIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
)

const CameraIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
)

const TrashIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/>
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
    <path d="M10 11v6M14 11v6"/>
    <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
  </svg>
)

const LeafIcon = ({ size = 40 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/>
    <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
  </svg>
)

const RecycleIcon = ({ size = 40 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="1 4 1 10 7 10"/>
    <polyline points="23 20 23 14 17 14"/>
    <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
  </svg>
)

const CloseIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18"/>
    <line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
)

// ─── Loading dots animation teks ─────────────────────────────────────────────
function DetectingText() {
  const [dots, setDots] = useState('')
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(d => d.length >= 6 ? '' : d + '.')
    }, 300)
    return () => clearInterval(interval)
  }, [])
  return <span>Mendeteksi{dots}</span>
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [image, setImage]           = useState(null)
  const [result, setResult]         = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [showResult, setShowResult] = useState(false)
  const [showError, setShowError]   = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [tab, setTab]               = useState('upload')

  const fileInputRef = useRef(null)
  const videoRef     = useRef(null)
  const streamRef    = useRef(null)

  // ── Reset ────────────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    setImage(null)
    setResult(null)
    setError(null)
    setShowResult(false)
    setShowError(false)
    stopCamera()
  }, [])

  // ── File upload ──────────────────────────────────────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setResult(null); setError(null); setShowResult(false); setShowError(false)
    setImage({ file, url: URL.createObjectURL(file) })
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (!file || !file.type.startsWith('image/')) return
    setResult(null); setError(null); setShowResult(false); setShowError(false)
    setImage({ file, url: URL.createObjectURL(file) })
  }

  // ── Camera ───────────────────────────────────────────────────────────────────
  const cameraInputRef = useRef(null)

  const startCamera = () => {
    // Pakai native file input dengan capture — lebih reliable di semua HP
    cameraInputRef.current?.click()
  }

  const handleCameraCapture = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setResult(null); setError(null); setShowResult(false); setShowError(false)
    setImage({ file, url: URL.createObjectURL(file) })
    // Reset input agar bisa capture ulang
    e.target.value = ''
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    setCameraActive(false)
  }

  const capturePhoto = () => {
    if (!videoRef.current) return
    const canvas = document.createElement('canvas')
    canvas.width  = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0)
    canvas.toBlob((blob) => {
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
      setImage({ file, url: URL.createObjectURL(blob) })
      stopCamera()
      setResult(null); setError(null); setShowResult(false); setShowError(false)
    }, 'image/jpeg', 0.92)
  }

  // ── Predict ──────────────────────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!image?.file) return
    setLoading(true)
    setError(null)
    setResult(null)
    setShowResult(false)
    setShowError(false)

    try {
      const formData = new FormData()
      formData.append('file', image.file)

      // Jalankan API call dan timer 3 detik secara paralel
      const [res] = await Promise.all([
        axios.post(`${API_URL}/predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 30000,
        }),
        new Promise(resolve => setTimeout(resolve, 3000)), // minimum 3 detik loading
      ])

      setResult(res.data)
      setLoading(false)
      setShowResult(true)
    } catch (err) {
      const msg = err.code === 'ERR_NETWORK' || err.message?.includes('Network')
        ? 'Tidak dapat terhubung ke server. Pastikan backend FastAPI sudah berjalan di port 8000.'
        : err.response?.data?.detail || 'Terjadi kesalahan saat prediksi.'

      await new Promise(resolve => setTimeout(resolve, 3000))
      setError(msg)
      setLoading(false)
      setShowError(true)
    }
  }

  const isOrganic = result?.prediction === 'Organik'

  return (
    <div className={styles.app}>

      {/* ══════════════════════════════════════
          POPUP LOADING
      ══════════════════════════════════════ */}
      {loading && (
        <div className={styles.overlay}>
          <div className={styles.loadingPopup}>
            {/* Spinner ring */}
            <div className={styles.spinnerRing}>
              <div className={styles.spinnerTrack} />
              <div className={styles.spinnerArc} />
              <div className={styles.spinnerIcon}>🔍</div>
            </div>
            <p className={styles.loadingTitle}><DetectingText /></p>
            <p className={styles.loadingSubtitle}>Sedang menganalisis gambar sampah</p>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════
          POPUP HASIL
      ══════════════════════════════════════ */}
      {showResult && result && (
        <div className={styles.overlay} onClick={() => setShowResult(false)}>
          <div
            className={`${styles.resultPopup} ${isOrganic ? styles.resultPopupOrganic : styles.resultPopupRecyclable}`}
            onClick={e => e.stopPropagation()}
          >
            {/* Close button */}
            <button className={styles.closeBtn} onClick={() => setShowResult(false)}>
              <CloseIcon />
            </button>

            {/* Icon besar */}
            <div className={`${styles.resultBigIcon} ${isOrganic ? styles.bigIconOrganic : styles.bigIconRecyclable}`}>
              {isOrganic ? <LeafIcon size={48} /> : <RecycleIcon size={48} />}
            </div>

            {/* Label */}
            <p className={styles.resultPopupLabel}>Hasil Prediksi</p>

            {/* Nama kelas */}
            <h2 className={`${styles.resultPopupName} ${isOrganic ? styles.textOrganic : styles.textRecyclable}`}>
              {isOrganic ? 'Organik' : 'Anorganik'}
            </h2>

            {/* Deskripsi */}
            <p className={styles.resultPopupDesc}>
              {isOrganic
                ? 'Sampah organik — dapat dikompos atau dijadikan pupuk.'
                : 'Sampah anorganik — dapat didaur ulang kembali.'}
            </p>

            {/* Divider */}
            <div className={styles.resultDivider} />

            {/* Confidence */}
            <p className={styles.confidenceLabel}>Tingkat Keyakinan</p>
            <p className={`${styles.confidenceBig} ${isOrganic ? styles.textOrganic : styles.textRecyclable}`}>
              {result.confidence}%
            </p>
            <div className={styles.progressBarFull}>
              <div
                className={`${styles.progressFill} ${isOrganic ? styles.fillOrganic : styles.fillRecyclable}`}
                style={{ width: `${result.confidence}%` }}
              />
            </div>

            {/* Tombol tutup */}
            <button
              className={`${styles.btnClose} ${isOrganic ? styles.btnCloseOrganic : styles.btnCloseRecyclable}`}
              onClick={() => setShowResult(false)}
            >
              Tutup
            </button>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════
          POPUP ERROR
      ══════════════════════════════════════ */}
      {showError && error && (
        <div className={styles.overlay} onClick={() => setShowError(false)}>
          <div className={styles.errorPopup} onClick={e => e.stopPropagation()}>
            <button className={styles.closeBtn} onClick={() => setShowError(false)}>
              <CloseIcon />
            </button>
            <div className={styles.errorBigIcon}>⚠️</div>
            <h3 className={styles.errorTitle}>Terjadi Kesalahan</h3>
            <p className={styles.errorMsg}>{error}</p>
            <button className={styles.btnCloseError} onClick={() => setShowError(false)}>
              Tutup
            </button>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════
          HEADER
      ══════════════════════════════════════ */}
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>♻️</span>
            <div>
              <h1 className={styles.title}>Smart Waste Classifier</h1>
              <p className={styles.subtitle}>AI-powered waste classification system</p>
            </div>
          </div>
          <div className={styles.badge}>ML Project</div>
        </div>
      </header>

      {/* ══════════════════════════════════════
          MAIN
      ══════════════════════════════════════ */}
      <main className={styles.main}>
        <div className={styles.container}>

          {/* Input Card */}
          <div className={styles.card}>
            {/* Tabs */}
            <div className={styles.tabs}>
              <button
                className={`${styles.tab} ${tab === 'upload' ? styles.tabActive : ''}`}
                onClick={() => { setTab('upload'); stopCamera() }}
              >
                <UploadIcon /> Upload Gambar
              </button>
              <button
                className={`${styles.tab} ${tab === 'camera' ? styles.tabActive : ''}`}
                onClick={() => { setTab('camera'); setImage(null); setResult(null) }}
              >
                <CameraIcon /> Kamera
              </button>
            </div>

            {/* Upload Tab */}
            {tab === 'upload' && (
              <div
                className={`${styles.dropzone} ${image ? styles.dropzoneHasImage : ''}`}
                onClick={() => !image && fileInputRef.current?.click()}
                onDragOver={e => e.preventDefault()}
                onDrop={handleDrop}
              >
                {image ? (
                  <div className={styles.previewWrapper}>
                    <img src={image.url} alt="Preview" className={styles.preview} />
                    <button className={styles.removeBtn} onClick={e => { e.stopPropagation(); reset() }}>
                      <TrashIcon /> Hapus
                    </button>
                  </div>
                ) : (
                  <div className={styles.dropzoneContent}>
                    <div className={styles.dropzoneIcon}><UploadIcon /></div>
                    <p className={styles.dropzoneText}>Klik atau drag & drop gambar di sini</p>
                    <p className={styles.dropzoneHint}>Mendukung JPG, PNG, WEBP</p>
                  </div>
                )}
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className={styles.hiddenInput} />
              </div>
            )}

            {/* Camera Tab */}
            {tab === 'camera' && (
              <div className={styles.cameraSection}>
                {/* Hidden input untuk capture kamera native HP */}
                <input
                  ref={cameraInputRef}
                  type="file"
                  accept="image/*"
                  capture="environment"
                  onChange={handleCameraCapture}
                  className={styles.hiddenInput}
                />

                {!image && (
                  <div className={styles.cameraPlaceholder}>
                    <div className={styles.dropzoneIcon}><CameraIcon /></div>
                    <p className={styles.dropzoneText}>Ambil foto langsung dari kamera HP</p>
                    <p className={styles.dropzoneHint}>Akan membuka kamera bawaan perangkat</p>
                    <button className={styles.btnSecondary} onClick={startCamera}>
                      <CameraIcon /> Buka Kamera
                    </button>
                  </div>
                )}

                {image && (
                  <div className={styles.previewWrapper}>
                    <img src={image.url} alt="Captured" className={styles.preview} />
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button className={styles.btnSecondary} onClick={startCamera}>
                        <CameraIcon /> Foto Ulang
                      </button>
                      <button className={styles.removeBtn} onClick={reset}>
                        <TrashIcon /> Hapus
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Predict Button */}
            <button
              className={`${styles.btnPredict} ${(!image || loading) ? styles.btnDisabled : ''}`}
              onClick={handlePredict}
              disabled={!image || loading}
            >
              🔍 Prediksi Sampah
            </button>
          </div>

          {/* Info Cards */}
          <div className={styles.infoGrid}>
            <div className={`${styles.infoCard} ${styles.infoOrganic}`}>
              <div className={styles.infoIcon}><LeafIcon size={28} /></div>
              <h3>Organik</h3>
              <p>Sisa makanan, daun, ranting, kertas basah, dan bahan alami lainnya.</p>
            </div>
            <div className={`${styles.infoCard} ${styles.infoRecyclable}`}>
              <div className={styles.infoIcon}><RecycleIcon size={28} /></div>
              <h3>Anorganik</h3>
              <p>Plastik, logam, kaca, kertas kering, dan bahan yang dapat didaur ulang.</p>
            </div>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className={styles.footer}>
        <p>Smart Waste Classifier · Tugas Machine Learning · Informatika</p>
      </footer>
    </div>
  )
}
