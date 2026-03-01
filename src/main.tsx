import { StrictMode, lazy, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { createHashRouter, RouterProvider } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import LandingPage from './components/LandingPage.tsx'

const FieldNotes = lazy(() => import('./components/FieldNotes.tsx'))

const router = createHashRouter([
  { path: '/', element: <LandingPage /> },
  { path: '/visualizer', element: <App /> },
  {
    path: '/field-notes',
    element: (
      <Suspense fallback={<div className="min-h-screen bg-gray-950 flex items-center justify-center text-gray-400">Loading...</div>}>
        <FieldNotes />
      </Suspense>
    ),
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
