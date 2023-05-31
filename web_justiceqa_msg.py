
import os
import re
import json

from modules import gossip_robot,justice_robot,classifier
from utils.json_utils import dump_user_dialogue_context,load_user_dialogue_context

from flask import Flask, request
from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
'''
    WebSocket服务开启，完成/justiceqa页面的消息通信
'''
# 开启一个flask应用
app = Flask(__name__)

# 机器人
def text_replay(user,msg):
    user_intent = classifier(msg)
    # print("进行意图分类")
    if user_intent in ["greet","goodbye","deny","isbot"]:
        reply = gossip_robot(user_intent)
        # print("调用闲聊机器人")
    elif user_intent == "accept":
        reply = load_user_dialogue_context(user)
        reply = reply.get("choice_answer","")
    else:
        # print("准备调用司法机器人")
        reply = justice_robot(msg, user)
        # print("type of reply1:", type(reply))
        # print("reply1:", reply)
        # print("成功调用司法机器人")
        if reply["slot_values"]:
            dump_user_dialogue_context(user, reply)
        reply = reply.get("reply_answer", "")
        # print("type of reply2:", type(reply))
        # print("reply2:", reply)
        # print("返回司法答案")

    if '\n' in reply:
        reply = reply.strip().replace('\\n','<br>')
    return reply

# 定义路由和函数功能
@app.route('/msg')
def msg():
	# 接收连接用户socket
    user_socker = request.environ.get('wsgi.websocket')
    # print("接收连接用户socket")
    # 保持与客户端通信
    while 1:
    	# 接收客户端发来的消息
        msg = user_socker.receive()
        # print("msg:", msg)
        # print("接收客户端发来的消息")
        if str(msg) == '':
            continue
        reply = text_replay('default_user',msg)
        # print("reply3:", reply)
        # print("调用text_replay机器人，传入msg")
        # 将要返回给客户端的数据封装到一个字典
        res = {
            "id" : 0, 
            "user" : 'https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png',
            "msg" : reply
        }
        # print("将要返回给客户端的数据封装到一个字典，封装完毕")
        # 编码为json格式并发送给客户端
        user_socker.send(json.dumps(res))
        # print(json.dumps(res))
        # print("编码为json格式并发送给客户端")

# @app.route('/')
# def hello_world():
#     return render_template('test.html')

if __name__ == '__main__':
    # app.run()
	# 创建一个服务器，IP地址为0.0.0.0，端口是9687，处理函数是app
    http_server = WSGIServer(('0.0.0.0', 1111), app, handler_class=WebSocketHandler)
    # print("开启服务器")
    # 开始监听请求:
    http_server.serve_forever()
    # print("开始监听")
    # app.run()