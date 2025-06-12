import streamlit as st
import sys
import pathlib
import asyncio
from PIL import Image
from fastai.vision.all import *
 # 新增：解决嵌套事件循环问题

# 确保Python版本兼容
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

# 修复事件循环问题
 # 新增：启用嵌套事件循环支持

# 移除已废弃的 add_script_run_ctx 调用
# add_script_run_ctx(loop)  # 这行需要删除，因为 streamlit.scriptrunner 已废弃

@st.cache_resource
def load_model():
    try:
        model_path = pathlib.Path(__file__).parent / "dish.pkl"
        return load_learner(model_path)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

# 加载模型
model = load_model()

st.title("Doraemon 与 Walle 分类器")
st.write("上传一张图片，看看它是 Doraemon 还是 Walle！")

# 图片上传和处理
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # 使用PIL加载图片
        image = PILImage.create(uploaded_file)
        st.image(image, caption="上传的图片", use_column_width=True)
        
        # 检查模型是否成功加载
        if model is not None:
            pred, pred_idx, probs = model.predict(image)
            st.success(f"预测结果：{pred}")
            st.write(f"概率：{probs[pred_idx]:.04f}")
        else:
            st.warning("模型未成功加载，无法进行预测。")
    except Exception as e:
        st.error(f"预测过程中发生错误: {e}")