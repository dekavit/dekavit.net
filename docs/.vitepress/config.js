import { defineConfig } from 'vitepress'
import katex from 'markdown-it-katex'
export default defineConfig({
    lang: "ja",
    title: "dekavit.net",
    description: 'デフォルト説明文',
    themeConfig: {
        nav: [
            { text: "top", link: "/" },
            { text: "algorithm", link: "/algorithm/" },
        ],
        socialLinks: [
            { icon: "github", link: "https://github.com/dekavit/" },
        ],
    },
    markdown: {
        config: md => md.use(katex),
    },
})