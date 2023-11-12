import { defineConfig } from 'vitepress'
import katex from "markdown-it-katex"
// export default defineConfig({
//     lang: "ja",
//     title: "dekavit.net",
//             themeConfig: {
//                 nav: [
//                     { text: "top", link: "/dist/" },
//                 ],
//                 socialLinks: [
//                     { icon: "github", link: "https://github.com/dekavit/" },
//                 ],
//             },
    // base: '/dist/',
//         markdown: {
//             config: md => md.use(katex),
//     },
// })
export default {
    base: '/dist/',
    markdown: {
        config: md => md.use(katex),
    }
}