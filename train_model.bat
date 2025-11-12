@echo off
echo ========================================
echo   Banana Ripeness Model Training
echo ========================================
echo.

cd src
python train.py

echo.
echo ========================================
echo   Training Complete!
echo ========================================
echo.
echo Check the models/ folder for:
echo   - best_model.pth (best model)
echo   - training_history.png (curves)
echo   - confusion_matrix.png (results)
echo.
pause
