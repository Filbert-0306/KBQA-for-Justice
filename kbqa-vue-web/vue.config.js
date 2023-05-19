const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  devServer: {
    proxy: {
      '/': {//代理api
        target: "http://127.0.0.1:5000",// 代理接口(注意只要域名就够了)
        changeOrigin: true,//是否跨域
        ws: false, // proxy websockets
        secure:false,
      },
      // '/justiceqa?':{
      //   target: "http://127.0.0.1:5000",// 代理接口(注意只要域名就够了)
      //   changeOrigin: true,//是否跨域
      //   ws: false, // proxy websockets
      //   secure:false,
      // }
    }
  },
  transpileDependencies: true,
  outputDir: '../static',
  indexPath: '../templates/test.html',
  assetsDir: 'static/'
})
