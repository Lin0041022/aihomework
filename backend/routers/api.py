from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import shutil
import os
import traceback
import matplotlib.pyplot as plt

from backend.database.db import SessionLocal
from backend.services.warning_service import HousePriceAnalysisBusinessService

router = APIRouter()
house_analysis_service = HousePriceAnalysisBusinessService()  # 初始化业务服务层


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
        result = house_analysis_service.get_import_records_list(db)
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
        result = house_analysis_service.load_historical_data(import_id, db)
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

        result = house_analysis_service.load_new_data_from_file(temp_path, db)
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
        result = house_analysis_service.execute_data_preprocessing()
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
        result = house_analysis_service.build_prediction_model()
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型构建失败: {str(e)}")


@router.post("/analysis/generate")
def generate_house_analysis(db: Session = Depends(get_db)):
    """
    生成房价分析
    """
    try:
        result = house_analysis_service.generate_house_analysis(db)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成房价分析失败: {str(e)}")


@router.get("/visualizations")
def get_visualizations():
    """
    获取可视化图表（返回图像文件）
    """
    try:
        result = house_analysis_service.generate_visualization_data(save_image=True)
        if result["success"]:
            image_paths = result.get("image_paths", [])
            if image_paths and os.path.exists(image_paths[0]):
                return FileResponse(
                    image_paths[0],
                    media_type="image/png",
                    filename="house_price_analysis.png"
                )
            else:
                raise HTTPException(status_code=500, detail="图像文件生成失败")
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成可视化图表失败: {str(e)}")


@router.get("/export/analysis")
def export_analysis():
    """
    导出分析结果
    """
    try:
        result = house_analysis_service.export_analysis_results(export_format="excel")
        if result["success"]:
            export_path = result["file_path"]
            return FileResponse(
                export_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename="house_price_analysis.xlsx"
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出分析结果失败: {str(e)}")


@router.get("/house/{house_id}")
def get_house_details(house_id: str):
    """
    获取特定房屋的分析详情
    """
    try:
        result = house_analysis_service.get_house_analysis_details(house_id)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取房屋详情失败: {str(e)}")


@router.get("/visualizations/model-evaluation")
def get_model_evaluation_charts():
    """
    获取模型评估图表
    """
    try:
        result = house_analysis_service.generate_model_evaluation_charts(save_image=True)
        if result["success"]:
            image_paths = result.get("image_paths", [])
            if image_paths and os.path.exists(image_paths[0]):
                return FileResponse(
                    image_paths[0],
                    media_type="image/png",
                    filename="model_evaluation.png"
                )
            else:
                raise HTTPException(status_code=500, detail="模型评估图像生成失败")
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成模型评估图表失败: {str(e)}")


@router.get("/system/status")
def get_system_status():
    """
    获取系统状态
    """
    try:
        result = house_analysis_service.get_system_status()
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


