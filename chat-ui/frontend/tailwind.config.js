/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        databricks: {
          red: '#FF3621',
          orange: '#FF6F0F',
          dark: '#1B1B1B',
          darker: '#0D0D0D',
          gray: '#2D2D2D',
          light: '#E8E8E8',
        }
      },
      animation: {
        'pulse-dot': 'pulse-dot 1.4s infinite ease-in-out both',
      },
      keyframes: {
        'pulse-dot': {
          '0%, 80%, 100%': { transform: 'scale(0)' },
          '40%': { transform: 'scale(1)' },
        }
      }
    },
  },
  plugins: [],
}
