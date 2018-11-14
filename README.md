# kaggle-dog-cat-knn
## Build model
Để build model chạy file classifier.py với cú pháp như sau 
```
python classifier.py features/vgg16_fc2/train
```
Để load các features model và train cho mô hình phân lớp. Mô hình sau khi train sẽ được lưu vào model/knn.joblib

## Testing
Để test model chạy file test.py với cú pháp như sau 
```
python test.py features/vgg16_fc2/test1
```
Để load models trước đó, nạp các features trong bộ dữ liệu test và ghi kết quả phần lớp vào file result.csv
