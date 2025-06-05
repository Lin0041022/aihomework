from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import shutil
import os

from backend.database.db import SessionLocal
# from backend.services.warning_service import AcademicWarningBusinessService  # 引入业务服务层
from backend.services.warning_service import AcademicWarningBusinessService

router = APIRouter()
warning_service = AcademicWarningBusinessService()  # 初始化业务服务层


# -------------------- 依赖 -------------------- #
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------- 接口实现 -------------------- #

@router.get("/records/imports")
def get_import_records(db: Session = Depends(get_db)):
    """
    获取导入记录列表
    """
    try:
        result = warning_service.get_import_records_list(db)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取导入记录失败: {str(e)}")


@router.post("/data/load-history")
def load_history(import_id: str = Form(...), db: Session = Depends(get_db)):
    """
    加载历史数据
    """
    try:
        result = warning_service.load_historical_data(import_id, db)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"加载历史数据失败: {str(e)}")


@router.post("/data/load-new")
def load_new_data(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    加载新数据
    """
    try:
        temp_path = f"temp_uploads/{file.filename}"
        os.makedirs("temp_uploads", exist_ok=True)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = warning_service.load_new_data_from_file(temp_path, db)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载新数据失败: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # 清理临时文件


@router.post("/data/preprocess")
def preprocess_data():
    """
    执行数据预处理
    """
    try:
        result = warning_service.execute_data_preprocessing()
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据预处理失败: {str(e)}")


@router.post("/model/build")
def build_model():
    """
    构建预测模型
    """
    try:
        result = warning_service.build_prediction_model()
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型构建失败: {str(e)}")


@router.post("/warnings/generate")
def generate_warnings(db: Session = Depends(get_db)):
    """
    生成学业预警
    """
    try:
        result = warning_service.generate_academic_warnings(db)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成预警失败: {str(e)}")


# @router.get("/visualizations")
# def get_visualizations():
#     """
#     获取可视化数据
#     """
#     try:
#         result = warning_service.generate_visualization_data()
#         if result["success"]:
#             return result
#         else:
#             raise HTTPException(status_code=500, detail=result["message"])
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"生成可视化数据失败: {str(e)}")

@router.get("/visualizations")
def get_visualizations():
    """
    获取可视化图表（返回图像文件）
    """
    try:
        result = warning_service.generate_visualization_data(save_image=True)
        if result["success"]:
            image_path = result["visualization_data"].get("image_path")
            if image_path and os.path.exists(image_path):
                return FileResponse(
                    image_path,
                    media_type="image/png",
                    filename="academic_report.png"
                )
            else:
                raise HTTPException(status_code=500, detail="图像文件生成失败")
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成可视化图表失败: {str(e)}")


@router.get("/export/warnings")
def export_warnings():
    """
    导出预警结果
    """
    try:
        result = warning_service.export_analysis_results(export_format="csv")
        if result["success"]:
            export_path = result["export_info"]["file_path"]
            return FileResponse(
                export_path,
                media_type="text/csv",
                filename=result["export_info"]["file_name"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出预警失败: {str(e)}")


@router.get("/student/{student_id}")
def get_student_details(student_id: str):
    """
    获取特定学生的风险详情
    """
    try:
        result = warning_service.get_student_risk_details(student_id)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取学生详情失败: {str(e)}")
