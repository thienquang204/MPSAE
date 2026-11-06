Kế hoạch được chia thành 7 phần: thí nghiệm đã có, kiểm tra baseline, ablation phương pháp, phân tích representation, benchmark mới, pretrained models và thứ tự implementation.
I. Các thí nghiệm ImageNet hiện có
Thiết lập chung: ImageNet-1K, ResNet-18 và ResNet-50, dictionary size 8192, \(K_{\mathrm{train}}=8\), batch size 1024, Adam, 10 epochs, một training seed. Các phương pháp gồm CSR, CSR + annealing nội bộ, MPSAE và MPSAEv2.
- E01 — ImageNet 1-NN: Đánh giá accuracy bằng nearest-neighbor retrieval trên 50.000 validation images, sử dụng training features làm gallery.
- E02 — ImageNet linear probing: Đóng băng SAE; huấn luyện classifier tuyến tính riêng cho từng \(K\), đánh giá Top-1 accuracy trên ImageNet.
- E03 — Sparsity evaluation: Đo actual nonzero dimensions, empty-code ratio, dead features và khả năng bão hòa support.
- E04 — Runtime evaluation: Đo thời gian train SAE, encode embeddings và sparse retrieval.
- E05 — Reconstruction analysis: Đánh giá MSE tái tạo, so sánh với bias-only reconstruction, kiểm tra khả năng mất thông tin.
Trạng thái: Đã có kết quả ban đầu. E05 mới có diagnostic trên mẫu nhỏ, chưa có full-validation evaluation.
II. Các thí nghiệm xác minh kết quả hiện tại
Những thí nghiệm này cần thực hiện trước khi khẳng định MPSAE vượt CSR hoặc CSRv2.
- V01 — Dense-parent 1-NN: Đánh giá trực tiếp ResNet embeddings bằng 1-NN, không qua SAE, để có baseline đúng metric.
- V02 — CSR reproduction: Kiểm tra chênh lệch giữa CSR nội bộ và kết quả công bố: preprocessing, batch size, số bước train, gallery và cách train theo \(K\).
- V03 — Retrieval correctness: So sánh kết quả sparse retrieval với dense exact dot-product trên một tập nhỏ; kiểm tra mã rỗng, index và tie-breaking.
- V04 — Linear-probe convergence: Kiểm tra learning rate, regularization và số epoch của probe; bảo đảm các representation được đánh giá bằng classifier hội tụ phù hợp.
- V05 — Full reconstruction evaluation: Đo reconstruction MSE, actual nonzeros, code norm và tail energy trên toàn bộ validation.
- V06 — OT solver correctness: Kiểm tra feasibility, độ chính xác nghiệm, gradient và sự khác biệt activity bookkeeping giữa CSR và MPSAE.
Ưu tiên: V01–V05 trước. V06 thực hiện trước khi chạy ablation thay đổi OT objective.
III. Các ablation về phương pháp MPSAE
A. Reconstruction và contrastive baselines
- A00 — Reconstruction-only: Bỏ toàn bộ contrastive/OT loss; kiểm tra reconstruction đơn thuần có tạo ra representation tốt không.
- A01 — CSR-global: CSR gốc với contrastive pool 1024; baseline chính.
- A02 — CSR-group: CSR với contrastive group size 32, tương đương kích thước nhóm OT; kiểm tra liệu lợi ích đến từ việc giảm interaction pool.
- A03 — CSR + annealing: CSR với main \(K\) giảm từ 64 xuống 8; kiểm tra lợi ích curriculum. Đây là baseline annealing nội bộ, không phải supervised CSRv2-linear đầy đủ.
B. Cross-budget representation learning
- A04 — Cross-budget NCL: Dùng contrastive learning giữa \(K,2K,4K\), không dùng OT; kiểm tra liệu cross-budget contrastive đã đủ tốt.
- A05 — Independent-budget NCL: Học contrastive riêng tại \(K,2K,4K\), rồi lấy trung bình loss; kiểm tra có thực sự cần tương tác giữa các budget.
- A06 — Same-budget OT: Dùng three-marginal matching với ba view \((K,K,K)\); kiểm tra tác dụng của OT khi không có khác biệt sparsity.
- A07 — MPSAE: Partial three-marginal OT với \((K,2K,4K)\); phương pháp chính để so sánh.
C. Partial và multi-marginal optimal transport
- A08 — Equal-marginal OT: Giữ transported mass \(s=0.9\), nhưng buộc mọi mẫu có marginal \(s/n\); kiểm tra adaptive partial matching có cần thiết.
- A09 — Pairwise partial OT: Thay three-marginal OT bằng ba pairwise partial matching losses; kiểm tra liệu joint multi-marginal coupling có thật sự hữu ích.
- A10 — Uniform-plan: Thay optimized transport plan bằng uniform coupling, sử dụng surrogate gradient tương ứng; kiểm tra có cần thực hiện tối ưu OT.
- A11 — Product-marginal: Giữ marginal của optimized transport plan nhưng bỏ dependency giữa các mẫu; kiểm tra joint matching structure có đóng góp gì.
D. MPSAEv2 và annealing
- A12 — MPSAEv2: Main \(K=8\) cố định, chỉ anneal hai OT auxiliary views từ \((128,256)\) về \((16,32)\).
- A13 — Fixed-wide OT: Giữ auxiliary budgets \((128,256)\) suốt quá trình; kiểm tra lợi ích có đến từ wide views chứ không phải annealing.
- A14 — Shuffled curriculum: Giữ cùng các budget trong lịch MPSAEv2 nhưng xáo thứ tự; kiểm tra tác dụng của curriculum ordering.
- A15 — Reversed curriculum: Đảo ngược lịch học; kiểm tra chiều annealing có quan trọng.
Ablation nên chạy trước: A00, A02, A04, A05, A08, A09, A10; sử dụng A01 và A07 làm mốc so sánh.

Ablation có thể hoãn: A06, A11, A13–A15. Chỉ mở rộng nghiên cứu MPSAEv2 nếu A12 thực sự cải thiện ổn định so với A07.
IV. Các thí nghiệm phân tích representation
Phần lớn có thể tận dụng pretrained SAE checkpoints đã có.
- D01 — Decoder analysis: Giữ encoder đóng băng, thử decoder gốc, decoder điều chỉnh scale và decoder tuyến tính mới; kiểm tra thông tin có nằm trong code nhưng decoder không khai thác được không.
- D02 — Support-only representation: Giữ nguyên nonzero indices, thay giá trị thành 1, chạy lại 1-NN và linear probing.
- D03 — Coefficient sensitivity: Shuffle nonzero values trong support hoặc quantize FP16/INT8; kiểm tra vai trò của coefficient magnitude.
- D04 — Neighborhood preservation: Đo mức trùng lặp nearest neighbors và CKA giữa sparse representation và dense parent.
- D05 — Feature utilization: Đo dead features, phân phối activation, support saturation, effective number of features và tail energy.
- D06 — Gradient analysis: So sánh gradient norm và gradient cosine của reconstruction loss và OT loss.
- D07 — Reconstruction ablation: Bỏ lần lượt reconstruction tại \(K\), tại \(4K\), AuxK hoặc toàn bộ reconstruction loss.
- D08 — Gradient routing: Cho OT gradient đi qua main view, auxiliary views hoặc cả ba view; kiểm tra view nào có vai trò quan trọng.
- D09 — Training-budget generalization: Train tại \(K=2,8,32\), sau đó đánh giá nhiều \(K_{\mathrm{eval}}\); kiểm tra khả năng adaptive sparsity.
- D10 — Hyperparameter sensitivity: Thay đổi có chọn lọc dictionary size, batch size, OT group size, transport mass, entropy và loss coefficient.
- D11 — Efficiency and storage: Đo actual storage bytes, index size, encode latency, search latency và quality–storage trade-off.
- D12 — Multiple training seeds: Chạy ít nhất ba seed cho các kết quả chính; báo mean, standard deviation và tính ổn định.
Ưu tiên phân tích: D01–D05 và D11 trước. D07–D10 cần train lại SAE nên chỉ chạy khi các thí nghiệm trước đặt ra câu hỏi rõ ràng.
V. Ba benchmark Representation Learning mới
G1 — Cross-dataset classification và few-shot learning
Dataset 1: Oxford Flowers-102
- Link: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Quy mô: 102 classes; 1.020 train, 1.020 validation và 6.149 test images.
- Cần tải: Images, class labels và official train/val/test split.
- Đầu vào: Global embeddings từ frozen backbone và ImageNet-trained SAE.
- Linear probing: Train classifier tuyến tính trên official train, chọn hyperparameters trên validation, đánh giá test.
- Few-shot: Chọn 5 ảnh/class trong training split, lặp lại với 5 sampling seeds.
- Metric: Top-1 accuracy và mean-per-class accuracy.
- Mục tiêu: Kiểm tra khả năng chuyển giao thông tin phân loại sang fine-grained categories.
- Train thêm: Chỉ linear classifier, không train lại SAE.
Dataset 2: Describable Textures Dataset (DTD)
- Link: https://www.robots.ox.ac.uk/~vgg/data/dtd/
- Quy mô: 5.640 ảnh, 47 texture classes.
- Cần tải: Images và 10 official train/val/test splits.
- Linear probing: Train trên 40 ảnh/class, chọn hyperparameters bằng validation, đánh giá 10 ảnh/class trong test.
- Few-shot: Chọn 5 ảnh/class từ training partition.
- Metric: Top-1 accuracy, mean ± std theo 10 official splits; báo biến thiên few-shot seed riêng.
- Mục tiêu: Kiểm tra MPSAE có giữ texture information, không chỉ object-category information.
- Train thêm: Chỉ linear classifier.
Cấu hình G1 chung: \(K_{\mathrm{eval}}=\{2,8,32\}\), frozen backbone, frozen SAE, cùng giao thức linear probe.
G2 — Semantic segmentation bằng frozen patch features
Dataset: PASCAL VOC 2012
- Link: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/
- Quy mô: 1.464 train và 1.449 validation images có segmentation annotations.
- Cần tải: RGB images, segmentation masks và official split files.
- Backbone: DINOv2 hoặc SigLIP 2 có spatial patch tokens.
- SAE: Patch-level SAE đã học từ ImageNet patch features.
- Input: Patch embeddings sau khi nén bằng MPSAE tại \(K=2,8,32\).
- Classifier: Một linear segmentation head với 21 output classes.
- Training: Đóng băng backbone và SAE; chỉ train linear head.
- Evaluation: Upsample predictions lên kích thước mask; ignore label 255.
- Metric chính: mIoU.
- Metric phụ: Patch 1-NN segmentation sử dụng labeled patch memory từ training set.
- Mục tiêu: Kiểm tra sparse features có giữ thông tin ngữ nghĩa tại từng vùng ảnh không.
- Train thêm: Patch-SAE nếu chưa có, sau đó chỉ linear segmentation head.
G3 — Semantic correspondence
Dataset: SPair-71k
- Link: https://cvlab.postech.ac.kr/research/SPair-71k/
- Quy mô: Khoảng 1.800 ảnh, với 12.234 official test pairs.
- Cần tải: Images, keypoint annotations, image-pair lists và bounding boxes.
- Backbone: DINOv2 hoặc SigLIP 2.
- SAE: Dùng lại patch-SAE của G2.
- Input: Sparse patch feature maps của hai ảnh.
- Matching: Lấy descriptor tại source keypoint, tìm điểm tương ứng gần nhất trong target feature map bằng cosine similarity.
- Metric: PCK@0.1 theo target bounding box, thêm kết quả theo object category/viewpoint.
- Mục tiêu: Kiểm tra sparse representation giữ được local semantic correspondence không.
- Train thêm: Không, nếu patch-SAE đã có.
Chú ý: G2 và G3 không thể được xem là sử dụng đúng cùng global-SAE checkpoint RN50/RN18 hiện có. Chúng cần một patch-SAE tương thích, được train riêng trên ImageNet patch descriptors.
VI. Ba benchmark Black-box Retrieval
Cả ba benchmark dưới đây đều có thể chạy bằng global SAE pretrained weights trên ImageNet, không cần train lại SAE nếu backbone và feature preprocessing khớp checkpoint.
R1 — Stanford Online Products (SOP)
- Dataset: Stanford Online Products.
- Link: https://cvgl.stanford.edu/projects/lifted_struct/
- Quy mô: 59.551 training images, 60.502 test images.
- Cần tải: Images, Ebay_test.txt và product identity labels.
- Query/gallery: Dùng test partition; mỗi ảnh là query, tìm trong các test ảnh khác và loại self-match.
- Embedding: Dense parent, CSR, MPSAE, MPSAEv2.
- Sparsity: \(K=\{2,4,8,16,32\}\).
- Metric: Recall@1, Recall@10, Recall@100 và mAP@R.
- Mục tiêu: Kiểm tra khả năng nhận diện đúng product identity.
- Train thêm: Không.
- Ưu tiên: Cao nhất trong nhóm retrieval.
R2 — DeepFashion In-Shop Retrieval
- Dataset: DeepFashion In-Shop.
- Link: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html
- Quy mô: 14.218 query images và 12.612 gallery images.
- Cần tải: Images, identity labels và list_eval_partition.txt.
- Query/gallery: Sử dụng đúng official query/gallery partition.
- Embedding: Global ImageNet-trained SAE.
- Sparsity: \(K=\{2,4,8,16,32\}\).
- Metric: Recall@1, Recall@10 và Recall@20.
- Mục tiêu: Kiểm tra retrieval khi cùng một sản phẩm quần áo có pose, scale và appearance khác nhau.
- Train thêm: Không.
R3 — Revisited Oxford / Paris
- Dataset: ROxford5k và RParis6k.
- Dataset link: https://cmp.felk.cvut.cz/revisitop/
- Evaluation code: https://github.com/filipradenovic/revisitop
- Quy mô: 70 queries cho Oxford và 70 queries cho Paris.
- Cần tải: Oxford/Paris images và gnd_roxford5k.pkl, gnd_rparis6k.pkl.
- Input: Sử dụng official query crops và toàn bộ gallery.
- Embedding: Global ImageNet-trained SAE.
- Sparsity: \(K=\{2,4,8,16,32\}\).
- Metric: mAP Medium và mAP Hard.
- Mục tiêu: Kiểm tra domain transfer sang landmark retrieval, viewpoint variation và occlusion.
- Train thêm: Không.
Ba retrieval ablations bổ sung
- RA1 — Objective ablation: Trên SOP, so sánh MPSAE với CSR-group, cross-budget NCL, pairwise OT và equal-marginal OT. Cần train các SAE variants trên ImageNet.
- RA2 — Sparse-code information: Trên SOP và DeepFashion, so sánh original code, support-only, shuffled coefficients và quantized coefficients. Không cần train lại SAE.
- RA3 — Retrieval efficiency: Trên SOP đo bytes/embedding, index-building time, encoding latency và search latency; trình bày đồ thị Recall–Storage và Recall–Latency.
VII. Pretrained models và weights cần chuẩn bị
1. ResNet-50
- Model: RAIVNLab/MRL, checkpoint r50_mrl0_e0_ff2048.pt.
- Link: https://github.com/RAIVNLab/MRL
- Feature dimension: 2048.
- Cần: Pretrained backbone weights, đúng preprocessing và SAE RN50 checkpoints đã train.
- Trạng thái: Đã có theo tài liệu dự án.
- Benchmark chạy được ngay: G1, R1, R2, R3.
2. ResNet-18
- Model: timm/resnet18.a1_in1k.
- Link: https://huggingface.co/timm/resnet18.a1_in1k
- Feature dimension: 512.
- Cần: Pretrained backbone weights, đúng preprocessing và SAE RN18 checkpoints.
- Trạng thái: Đã có theo tài liệu dự án.
- Benchmark chạy được ngay: G1, R1, R2, R3.
3. DINOv2 ViT-B/14
- Model: dinov2_vitb14.
- Link: https://github.com/facebookresearch/dinov2
- Global features: CLS token, dimension 768.
- Patch features: Final patch tokens, dimension 768.
- Cần: Pretrained DINOv2 weights, ImageNet RGB images và hai SAE checkpoints (global/patch).
- Trạng thái: Cần train SAE riêng nếu chưa có.
- Benchmark: Toàn bộ G1–G3 và R1–R3.
4. SigLIP 2 ViT-B/16
- Model: google/siglip2-base-patch16-224.
- Link: https://huggingface.co/google/siglip2-base-patch16-224
- Global features: Chọn cố định pooled/projection output.
- Patch features: Spatial tokens từ vision tower.
- Cần: Pretrained SigLIP 2 weights và SAE riêng cho global/patch features.
- Trạng thái: Cần train SAE riêng nếu chưa có.
- Benchmark: Toàn bộ G1–G3 và R1–R3.
5. Franca ViT-B/14
- Repository: https://github.com/valeoai/Franca
- Model đề xuất: franca_vitb14, cấu hình pretrained IN21K_224 không dùng RASA.
- Feature dimension: 768 với cấu hình ViT-B/14 được đề xuất.
- Cần: Pretrained backbone weights và SAE tương ứng.
- Trạng thái: Tùy chọn, dùng để xác nhận kết quả trên họ pretrained model thứ ba.
- Benchmark ưu tiên: G2, G3 và R1.
Nếu mới có backbone weights mà chưa có SAE: lấy 128.000 ImageNet training images để trích global features; với patch SAE, lấy 16.000 ảnh và sample 8 patches/ảnh. Train CSR và MPSAE trên cùng feature caches, rồi sử dụng các SAE này cho toàn bộ downstream benchmarks. Đây là cấu hình giảm dữ liệu đề xuất, không phải tái lập full-ImageNet benchmark.
VIII. Implementation plan

Phase 0 — Verify existing ImageNet results

Chạy V01–V05, kiểm tra CSR baseline, dense-parent retrieval, linear probes và reconstruction. Không cần train lại SAE trừ khi cần tái lập baseline.




Phase 1 — Evaluate existing SAE weights

Download Flowers, DTD và SOP. Cache RN50/RN18 features, chạy G1 và R1. Tiếp tục R2, R3 sau khi xác nhận evaluator đúng. Không train lại SAE.




Phase 2 — Ablation phương pháp

Chạy A00, A02, A04, A05, A08, A09, A10 trên một backbone trước; xác nhận kết quả quan trọng trên SOP và ImageNet. Bắt đầu bằng pilot nhỏ, chỉ full-fit những variants cần thiết.




Phase 3 — Modern pretrained backbones

Trích xuất ImageNet features bằng DINOv2 và SigLIP 2, train global CSR/MPSAE, đánh giá lại G1 và R1. Kiểm tra representation transfer qua các backbone khác nhau.




Phase 4 — Spatial representation

Train patch-SAE trên ImageNet patch features. Chạy G2 VOC segmentation và G3 SPair correspondence.




Phase 5 — Final validation

Xác nhận kết quả chính trên ít nhất ba training seeds; thực hiện RA2, RA3, bổ sung MPSAEv2 hoặc Franca nếu có lợi ích rõ ràng.



IX. Những điều bắt buộc giữ nhất quán
- Frozen-backbone protocol: Không cập nhật vision backbone ở bất kỳ benchmark chính nào.
- Source-only SAE training: Train SAE trên ImageNet features, không train SAE trên downstream test datasets.
- Same feature extraction: CSR, MPSAE và MPSAEv2 phải sử dụng cùng backbone output, preprocessing và split.
- Same retrieval protocol: Cùng normalization, distance metric, gallery/query và tie-breaking.
- Same evaluation budget: Các linear probes có cùng search budget và được kiểm tra convergence.
- Sparsity accounting: Báo cả requested \(K\), actual nonzeros và bytes gồm indices/values.
- Statistical reliability: Nhiều training seeds cho các kết quả chính; tách biến thiên của probe/few-shot sampling.
- Compute: Báo GPU-hours và thời gian encode/search riêng; không suy ra phương pháp nhanh hơn chỉ từ số nonzero.
X. Các kết quả cần lưu khi implementation
- run_config.json — Model, SAE, optimizer, seed, \(K\), objective và lịch training.
- source_manifest.json — Backbone checkpoint, feature layer, preprocessing và checksum.
- split_ids.csv — Danh sách image IDs của từng partition.
- metrics.csv — Dataset, method, backbone, seed, \(K\), actual nnz và kết quả metric.
- rankings/ — Retrieval rankings và predictions để tái lập Recall/mAP.
- runtime.json — Training, encoding, index-building, search latency và GPU memory.
- diagnostics.json — Empty codes, support distribution, reconstruction và OT solver diagnostics.

Thứ tự mình đề xuất chốt để triển khai
Nhóm bắt buộc: V01–V05, G1 (Flowers/DTD), R1 (SOP), A02, A04, A05, A08, A09, A10.

Nhóm rất nên có cho CVPR: R2, R3, D01–D05, D11, D12, DINOv2 global transfer, G2 và G3 nếu định claim phương pháp học representation không gian.

Nhóm chỉ làm khi có kết quả hỗ trợ: MPSAEv2 curriculum ablations, Franca, dictionary-width sweeps, nhiều lịch annealing, hoặc các hyperparameter sweeps lớn.

Với những checkpoints hiện tại, bước implementation có hiệu quả nhất là hoàn thành V01–V05, rồi G1 và R1 trước. Các kết quả đó sẽ cho chúng ta biết liệu MPSAE thực sự có khả năng chuyển giao ngoài ImageNet hay không, trước khi tiếp tục tiêu tốn GPU để train các model DINOv2, SigLIP 2 hoặc patch-SAE.