/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          green: "#8f9d68",   // Primary active highlight
          yellow: "#ffd700",  // Primary accent (used rarely)
          lightBg: "#f8fafc", // Light background
          card: "#ffffff",    // Card background
          border: "#e2e8f0",  // Light border
          header: "#d1d5db",  // Header grey background as shown in wireframe
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
