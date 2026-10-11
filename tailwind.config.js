/** @type {import('tailwindcss').Config} */
module.exports = {
  // Class names are scanned from the Jinja templates and the Alpine/vanilla JS.
  // Keep every class name a complete literal string so the JIT can find it.
  content: ["./app/templates/**/*.html", "./app/static/js/**/*.js"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "SFMono-Regular", "ui-monospace", "Menlo", "Consolas", "monospace"],
      },
      colors: {
        // Deep studio surfaces (ElevenLabs-ish: near-black slate + violet glow)
        surface: {
          950: "#050713",
          900: "#0b0f1f",
          850: "#111629",
          800: "#171d33",
          700: "#232a44",
        },
      },
      boxShadow: {
        panel: "0 30px 70px -40px rgba(0, 0, 0, 0.95)",
        glow: "0 0 0 1px rgba(139, 92, 246, 0.28), 0 18px 45px -18px rgba(99, 102, 241, 0.55)",
        "glow-soft": "0 0 40px -12px rgba(139, 92, 246, 0.45)",
      },
      backgroundImage: {
        "studio-grid":
          "linear-gradient(rgba(148,163,184,0.055) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.055) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "56px 56px",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(0.9)", opacity: "0.7" },
          "70%": { transform: "scale(1.6)", opacity: "0" },
          "100%": { transform: "scale(1.6)", opacity: "0" },
        },
        drift: {
          "0%, 100%": { transform: "translate3d(0,0,0)" },
          "50%": { transform: "translate3d(0,-18px,0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.35s ease both",
        "pulse-ring": "pulse-ring 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        drift: "drift 9s ease-in-out infinite",
        shimmer: "shimmer 2.2s linear infinite",
      },
    },
  },
  plugins: [],
};
