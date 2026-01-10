/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                midnight: '#0B1120',
                copper: {
                    400: '#F59E0B',
                    500: '#D97706',
                    glow: 'rgba(245, 158, 11, 0.1)',
                },
                rose: {
                    400: '#FB7185',
                    500: '#F43F5E',
                    glow: 'rgba(251, 113, 133, 0.1)',
                },
                emerald: {
                    400: '#34D399',
                    500: '#10B981',
                    glow: 'rgba(52, 211, 153, 0.1)',
                }
            },
            fontFamily: {
                sans: ['Plus Jakarta Sans', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
            },
            backgroundImage: {
                'copper-gradient': 'linear-gradient(to right, rgba(249, 115, 22, 0.1), rgba(244, 63, 94, 0.1))',
            },
            boxShadow: {
                'glow-copper': '0 0 30px -10px rgba(245, 158, 11, 0.3)',
                'glow-rose': '0 0 30px -10px rgba(244, 63, 94, 0.3)',
                'glow-emerald': '0 0 30px -10px rgba(52, 211, 153, 0.3)',
            }
        },
    },
    plugins: [],
}
