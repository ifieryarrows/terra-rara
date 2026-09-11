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
                    300: '#F4C5A5',
                    400: '#E6A47A',
                    500: '#C78155',
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
                sans: [
                    'Geist Sans',
                    'ui-sans-serif',
                    'system-ui',
                    '-apple-system',
                    'BlinkMacSystemFont',
                    'Segoe UI',
                    'Roboto',
                    'Helvetica Neue',
                    'Arial',
                    'Noto Sans',
                    'sans-serif',
                    'Apple Color Emoji',
                    'Segoe UI Emoji',
                    'Segoe UI Symbol',
                    'Noto Color Emoji',
                ],
                // Keep the utility available for existing metric markup, but
                // resolve it to the same family as the rest of the product.
                mono: [
                    'Geist Sans',
                    'ui-sans-serif',
                    'system-ui',
                    'sans-serif',
                ],
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
