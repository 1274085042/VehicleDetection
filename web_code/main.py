from flask import Flask
from flask import request, abort, send_file, render_template
import base64

from tf_serving_client import make_prediction


app = Flask(__name__)


@app.route("/api/v1/prediction/vehicle", methods=['POST'])
def commodity_predict():
    """
    商品图片预测REST接口
    """
    # 获取用户上传图片
    image = request.files.get('image')
    if not image:
        abort(400)

    # 预测标记
    result_img = make_prediction(image.read())
    data = result_img.read()
    result_img.close()

    return data, 200, {'Content-Type': 'image/png'}


@app.route("/api/v2/prediction/vehicle", methods=['POST'])
def commodity_predict_weixin():
    """
    商品图片预测REST接口
    """
    # 获取用户上传图片
    image = request.files.get('image')
    if not image:
        abort(400)

    # 预测标记
    result_img = make_prediction(image.read())
    data = result_img.read()
    result_img.close()

    b64 = base64.b64encode(data)

    return b64, 200, {'Content-Type': 'text/plain'}


@app.route("/")
def index():
    """
    Web页面
    """
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
