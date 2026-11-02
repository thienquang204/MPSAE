# Thống kê code và ý tưởng MMRSAE, MPSAE, CSR

> **Benchmark update, 2 November 2026:** The active comparison now contains
> only CSR, CSRv2, DLMMSAE and DLMMSAEv2 (`csr,csrv2,dlmmsae,dlmmsaev2`).
> DLMMSAEv2 replaces the name `dlmmsae-annealed` and uses CSRv2's exact K
> schedule. The inventory and workload counts below describe earlier ablations.
> See [MMRSAE_IMAGENET.md](MMRSAE_IMAGENET.md) for the current benchmark.

> **Architecture update, 2 November 2026:** Current MMRSAE now synthesizes a
> landmark-context embedding before tied SAE/Top-K, with no post-SAE repair.
> The fixed input landmark dictionary remains in the inference model. Clean
> supports are nested. MMRSAEv2 (`mmrsae-annealed`) uses actual training-K
> half-cosine annealing and B=4K(t), replacing its old auxiliary-only schedule.
> The sparse-start architecture and line references below are historical.
> The staged `repair` study retains sparse-start repair. See
> [MMRSAE_IMAGENET.md](MMRSAE_IMAGENET.md) for landmark-first v3.

> **Snapshot trước bản cập nhật tiếp theo ngày 02/11/2026.** Code hiện tại đã
> chuyển các objective MMRSAE/repair sang angular-only OT, thêm CSRv2 vào bộ so
> sánh và đánh giá/plot nhiều inference K trên cùng checkpoint. Các mô tả mixed
> OT, danh sách 6 phương pháp, số dòng và số test bên dưới là thông tin của lần
> kiểm kê trước thay đổi. Xem `MMRSAE_IMAGENET.md` để dùng protocol hiện tại.

Rà soát ngày 02/11/2026, dựa trên mã nguồn trong thư mục `graduate_thesis`. Các số dòng dưới đây là vị trí tại thời điểm rà soát. Đây là bản đồ cài đặt, không phải báo cáo kết quả huấn luyện.

## 1. Phạm vi và số lượng

Đếm trực tiếp các file Python ở thư mục gốc và `tests/`, các shell script ở gốc và hai file trong `csrc/`. Không tính thư viện trong môi trường ảo, cache, embedding, `__pycache__`, `.git` hoặc tài liệu HTML của các bài báo. Số dòng là số dòng vật lý, bao gồm dòng trống, comment và docstring.

| Thành phần | Số lượng |
|---|---:|
| Python ở thư mục gốc | 31 file |
| Python trong `tests/` | 6 file |
| Tổng Python | 37 file, 22.722 dòng |
| Class trong các file Python trên | 66 |
| Hàm/method, kể cả hàm lồng nhau | 781 |
| Shell script | 11 file, 1.735 dòng |
| C++/CUDA | 2 file, 152 dòng |
| Method kiểm thử tên `test_*` trong `tests/` và `check_reference.py` | 87 |

Số lượng này bao gồm cả runner, dữ liệu, báo cáo, transfer và thí nghiệm dense cũ; không phải toàn bộ đều là thuật toán mới. Rà soát này không chạy các test, chương trình huấn luyện, download dữ liệu hay benchmark CUDA. Không đọc file `.env` riêng tư và không thay đổi code thực thi.

## 2. Quan hệ giữa ba phương pháp

Cả ba phương pháp sparse đều nén embedding của backbone đã đóng băng bằng sparse autoencoder có trọng số encoder/decoder dùng chung.

| Nội dung | CSR | MPSAE | MMRSAE mặc định |
|---|---|---|---|
| Biểu diễn | Nonnegative Top-K trong dictionary rộng h | Giống CSR | Top-K sau một bước sửa từ residual |
| Reconstruction K và 4K | Có | Có | K sau repair; auxiliary raw mặc định 4K |
| Dead-latent auxiliary | Có | Có | Có |
| Nonnegative contrastive loss, NCL | Có | Không | Không |
| Matching gap OT ba marginal | Không | K/2K/4K | K đã repair / 2K đã repair / raw 4K |
| Mask hệ số khởi đầu | Không | Không | Có, trong training |
| KL theo landmark của teacher | Không | Không | Có |
| Cost OT có thông tin lân cận teacher | Không | Không | Có, trộn với angular cost |
| Thành phần cần khi inference | SAE | SAE | SAE và repair |

Phần khác biệt chính của MPSAE nằm ở objective: thay NCL của CSR bằng partial OT matching gap giữa ba mức sparsity. MMRSAE bổ sung khả năng thay đổi support bằng residual repair và học cấu trúc lân cận của embedding gốc.

Có nhiều đường chạy dùng cùng tên phương pháp. `csr`/`mpsae` trong runner MMRSAE là baseline dưới protocol MMRSAE; kết quả của chúng không tự động trở thành kết quả strict paper reproduction.

## 3. Lõi SAE và reconstruction dùng chung

Đặt `x ∈ R^d`, dictionary `D ∈ R^(h×d)`, bias `b_pre ∈ R^d`, `b_enc ∈ R^h`:

```text
a(x) = (x - b_pre) Dᵀ + b_enc
z_K  = H_K⁺(a(x))
x̂_K = z_K D + b_pre
```

`H_K⁺` chọn Top-K trên giá trị có dấu rồi áp dụng ReLU. Vì vậy code có **tối đa K phần tử dương**, không bảo đảm đúng K. Tọa độ phải được giữ cùng với giá trị; K giá trị không có indices không thể biểu diễn cùng sparse code.

| Khối code | Vị trí | Vai trò |
|---|---|---|
| `TopKSAE` | [csr_vs_mmpot_imagenet.py](csr_vs_mmpot_imagenet.py), dòng 1334 | SAE của các thí nghiệm ImageNet CSR/MPSAE v1–v2; transfer cũng import class này |
| `ResearchSAE` | [mpsae_repair.py](mpsae_repair.py), dòng 35 | SAE dùng bởi repair study và MMRSAE |
| `nonnegative_topk` | `mpsae_repair.py`, dòng 18 | Top-K có quy tắc hòa: ưu tiên tọa độ nhỏ; không dùng straight-through estimator cho lựa chọn support |
| `indexed_decode` | `mpsae_repair.py`, dòng 94 | Sparse COO matrix multiplication để decode mã ít phần tử hoạt động |
| `reconstruction_losses` | `csr_vs_mmpot_imagenet.py`, dòng 1385 | Reconstruction dùng cho MPSAE cũ |
| `csr_losses` | `csr_vs_mmpot_imagenet.py`, dòng 1431 | Reconstruction dùng cho CSR cũ |
| `reconstruction_terms` | `mpsae_repair.py`, dòng 173 | Reconstruction chung, có thể thêm repair/masking và đổi auxiliary budget |

Reconstruction mặc định:

```text
R = MSE(x̂_K, x) + (1/8) MSE(x̂_4K, x) + (1/32) L_dead
```

`L_dead` dùng các latent không hoạt động để reconstruct residual, rồi chuẩn hóa MSE theo baseline dự đoán bằng trung bình target. `k_aux=512`, ngưỡng dead là `inactive_steps > 30`. Counter được cập nhật theo các lần forward theo quy ước cài đặt, không phải số epoch.

Không có reconstruction riêng cho 2K trong base objective. 2K đóng vai trò view cho OT. Với MMRSAE, reconstruction chính dùng code K đã repair; nhánh auxiliary dùng raw code với budget B, mặc định B=4K.

Các cài đặt khởi tạo dictionary bằng Kaiming uniform, hai bias bằng 0; dictionary được dùng chung cho encode/decode. Những vòng train sparse được rà soát không áp dụng bước renormalize dictionary sau mỗi optimizer update.

## 4. CSR: reconstruction và nonnegative contrastive learning

```text
L_CSR = R + gamma × L_NCL
```

NCL tạo hai bản sao xác định của cùng code `z_K`, nối thành 2B mẫu, L2-normalize rồi tính logits cosine/temperature. Positive là hai bản sao cùng mẫu; các mẫu khác là negatives; loại self-pair trên đường chéo. Hai view này không phải hai ảnh augmentation được tạo mới trong SAE loop.

| Hàm | Vị trí |
|---|---|
| NCL của runner cũ | `csr_vs_mmpot_imagenet.py:1479` |
| Train CSR/CSRv2 | `csr_vs_mmpot_imagenet.py:1871`, `train_csr` |
| NCL của API MMRSAE | [mmrsae.py](mmrsae.py), dòng 42 |
| CSR baseline dưới protocol MMRSAE | `mmrsae.py:315`, `MMRSAE._csr_loss` |

Gamma mặc định là 0,1 với FF2048/ResNet-50 và 1,0 với nghiên cứu ResNet-18; temperature là 0,2. API MMRSAE suy ra gamma mặc định từ input dimension: 2048 → 0,1; dimension khác → 1,0.

CSR baseline trong MMRSAE dùng SAE thuần, không repair, không mask, không landmark KL và không OT. Counter hoạt động được cập nhật theo thứ tự K, 4K, K để khớp đường CSR.

## 5. MPSAE: partial matching giữa K, 2K và 4K

```text
L_MPSAE = R + lambda_OT × G_partial
```

Mặc định `lambda_OT=2`. Ba view là ba code từ cùng embedding, không phải ba modality. Trong MPSAE nguyên bản, chúng cùng lấy từ một preactivation nên raw support có tính lồng nhau, theo quy tắc chọn Top-K.

Với các vector đã chuẩn hóa `u_i`, `v_j`, `w_l` ở ba budget:

```text
C0[i,j,l] = (2/9) × [(1-u_i·v_j) + (1-u_i·w_l) + (1-v_j·w_l)]
```

Code clamp angular/circular-variance cost vào [0,1]. Tuple `(i,i,i)` biểu diễn cùng ảnh ở ba budget; tuple khác có thể ghép code từ nhiều ảnh.

Partial transport tìm tensor `P ≥ 0`, tổng mass bằng `s`, marginal từng phía không vượt capacity `1/n` mỗi mẫu. Phần mass không ghép được biểu diễn bằng slack. Reference matching đặt `s/n` lên các tuple đường chéo. Gap so sánh objective có entropy của reference với objective của transport tìm được.

Solver chạy với cost đã detach; gradient của gap theo cost có dạng `sJ - P*`, theo envelope/Danskin. Entropy được giữ trong giá trị loss được log nhưng không backprop qua các iteration của solver.

| Khối | Runner ImageNet cũ | Lõi hiện tại |
|---|---|---|
| Circular variance | `csr_vs_mmpot_imagenet.py:1503` | `mpsae_repair.py:399` |
| Greenkhorn ba marginal | `csr_vs_mmpot_imagenet.py:1521` | `mpsae_repair.py:481` |
| Solver batched | Không có bản riêng trong runner này | `mpsae_repair.py:538` |
| Matching gap | `partial_matching_gap`, dòng 1583 | `matching_gap_from_cost`, dòng 596 |
| Chia nhóm OT và ghép loss | `partial_matching_gap` | `mixed_matching_gap`, dòng 658 |
| Vòng train | `train_mp_sae`, dòng 2058 | `research_loss`, dòng 745, được gọi bởi `MMRSAE.loss` |

Mặc định `s=0,9`, entropy coefficient `eta=0,2`, 100 iteration ban đầu, tolerance `1e-4`, microbatch=32. Tensor cost của một nhóm 32 mẫu có 32³=32.768 phần tử. Matching gap được lấy trung bình trên các nhóm độc lập; nhóm đuôi có dưới hai mẫu bị bỏ qua bởi helper OT.

Lõi mới kiểm tra cost hợp lệ, mass và capacity; có bounded refinement tối đa 2.000 iteration khi cần. Nếu gap âm vượt tolerance, có thể solve lại trước khi báo lỗi; không che lỗi bằng cách clamp gap thành 0. Runner cũ có solver riêng và không có toàn bộ cơ chế kiểm tra/refinement này.

MPSAE baseline trong MMRSAE giữ objective angular OT, không bật repair, mask hoặc landmark KL. Nó chia sẻ protocol huấn luyện/đánh giá hiện tại nhưng không gọi nguyên xi vòng train cũ.

## 6. MMRSAE: sửa code sparse và học lân cận của teacher

API chính là [mmrsae.py](mmrsae.py): `MMRSAEConfig` ở dòng 62 và `MMRSAE` ở dòng 174. Phần tính toán được chia sẻ với [mpsae_repair.py](mpsae_repair.py), không phải một lõi thuật toán độc lập hoàn toàn.

### 6.1 Một bước residual repair

```text
z0 = raw Top-K code, hoặc code đã mask trong training
r  = x - b_pre - z0 D
z1 = H_K⁺(z0 + alpha × r Dᵀ + ReLU(r U) V)
```

`RepairModule` ở `mpsae_repair.py:70`; `U` có shape d×rank và `V` có shape rank×h. Rank mặc định 32; U khởi tạo ngẫu nhiên bằng generator riêng, V khởi tạo bằng 0. Thành phần `alpha rDᵀ` là correction kiểu IHT; thành phần low-rank học thêm cách sửa từ residual.

Alpha là buffer cố định, không phải parameter được optimizer học. `estimate_alpha` ở dòng 109 ước lượng `0,9 / ||D||²` bằng power iteration. `backtrack_alpha` ở dòng 125 giảm alpha đến khi fixed-IHT không tăng reconstruction error trên slice training dùng để hiệu chỉnh. `MMRSAE.initialize_repair` ở `mmrsae.py:221` hiệu chỉnh một lần rồi checkpoint giá trị này. Kiểm tra ban đầu không bảo đảm bước repair đã học luôn giảm lỗi sau mọi update.

Train sửa hai view K và 2K; view 4K cho angular OT vẫn là raw. Inference bắt đầu bằng code sạch rồi sửa đúng một bước ở budget được yêu cầu. Support sau repair có thể khác raw support và không còn bắt buộc lồng nhau giữa các budget.

### 6.2 Mask hệ số để tạo bài toán phục hồi

`mask_start` ở `mpsae_repair.py:140`:

- Với xác suất 0,2, giữ nguyên cả hàng.
- Các hàng còn lại xóa từng hệ số dương độc lập với xác suất 0,25.
- Không rescale các hệ số còn lại; target embedding x giữ nguyên.
- Dùng RNG riêng, có thể lưu và khôi phục khi resume.

Mask tạo cơ hội cho repair phục hồi hoặc chọn lại tọa độ. Inference và refresh landmark luôn dùng code khởi đầu sạch.

### 6.3 Teacher/student neighborhood profiles và KL

`LandmarkBank` ở `mpsae_repair.py:277` lưu bank mặc định 1.024 embedding training. Teacher chính là embedding backbone đóng băng. Student key là code sạch K/2K, được refresh định kỳ và detach khỏi gradient.

`LandmarkBank.profiles`, dòng 329, tính phân phối softmax của cosine similarity tới cùng tập landmark: một phân phối teacher và hai phân phối student ở K, 2K. Temperature mặc định của cả teacher/student là 0,1. Landmark có ID nằm trong **hợp toàn minibatch** bị loại khỏi mọi profile của minibatch đó.

`relational_kl` ở dòng 392 lấy trung bình `KL(p_teacher || p_student)` của hai view. Teacher và student key được detach; gradient đi qua student query code. `indexed_cosines` ở dòng 262 tính similarity bằng tọa độ sparse, chia chunk landmark để giới hạn intermediate.

### 6.4 Cost lân cận và một transport plan chung

`neighborhood_cost` ở `mpsae_repair.py:405`:

```text
Cnbr[i,j,l] = 1 - sum_m (pK_i[m] × p2K_j[m] × pTeacher_l[m])^(1/3)
Cmix        = (1-rho) C0 + rho Cnbr
```

Đây là distance từ affinity ba phía trên phân phối landmark. Code chuẩn hóa lại trong log space, hỗ trợ log-probability `-inf` với support hợp lệ, giữ float64 cho reference checks và dùng ít nhất float32 trong phép tính chính. Affinity được clamp vào [0,1] trước phép trừ để tránh cost âm do sai số số học.

`rho=0,2` trong các preset mixed hiện tại: 0,8 angular + 0,2 neighborhood. Hai cost được trộn **trước một bài toán OT cho mỗi microbatch**; không solve hai plan độc lập rồi cộng hai gap. Gradient của cost lân cận tác động tới hai student profile; teacher profile được cố định.

### 6.5 Objective và các preset

```text
L_MMRSAE = MSE(x̂_repaired_K, x)
          + (1/8) MSE(x̂_raw_B, x)
          + (1/32) L_dead
          + lambda_rel × L_rel
          + lambda_OT × G_partial(Cmix)
```

`research_loss`, `mpsae_repair.py:745`, lắp ghép objective. `MMRSAE.loss`, `mmrsae.py:275`, chọn đường CSR riêng hoặc gọi lõi này.

| Preset | Repair/mask | KL weight | OT weight | rho | Auxiliary reconstruction B |
|---|---|---:|---:|---:|---|
| `mmrsae` | Có/có | 1 | 2 | 0,2 | min(h,4K) |
| `mmrsae-rel` | Có/có | 1 | 0 | 0 | min(h,4K) |
| `mmrsae-wide` | Có/có | 1 | 2 | 0,2 | min(h,max(64,8K)) |
| `mmrsae-annealed` | Có/có | 1 | 2 | 0,2 | Giảm từ wide xuống 4K trong 70% update đầu |
| `mpsae` | Không/không | 0 | 2 | 0 | min(h,4K) |
| `csr` | Không/không | 0 | Không dùng | 0 | min(h,4K), cộng NCL |

Đây là mặc định của preset, một số weight có thể được override. `auxiliary_budget`, dòng 159, dùng lịch tuyến tính có làm tròn cho MMRSAE annealed. Budget được phục vụ K vẫn cố định; raw 4K vẫn tồn tại cho angular OT. Ví dụ K=8: B mặc định 32, wide 64, annealed 64→32.

### 6.6 Số parameter từ cấu trúc mặc định

Với tied dictionary, SAE có `hd+h+d` parameter; repair thêm `rank×(d+h)`. Alpha và inactivity counters là buffers.

| Backbone/input | SAE, h=8192 | Repair, rank=32 | Tổng có repair |
|---|---:|---:|---:|
| ResNet-50, d=2048 | 16.787.456 | 327.680 | 17.115.136 |
| ResNet-18, d=512 | 4.203.008 | 278.528 | 4.481.536 |

Đây là phép đếm theo khai báo module, không bao gồm parameter backbone đóng băng và không phải số đo bộ nhớ runtime.

## 7. Hai kiểu annealing có ý nghĩa khác nhau

| Nhóm | Hàm | Đại lượng thay đổi |
|---|---|---|
| `csrv2`, `mpsaev2` cũ | `cosine_annealed_k`, `csr_vs_mmpot_imagenet.py:246` | Giảm chính training K bằng half-cosine, rồi giữ target K |
| `mmrsae-annealed` | `auxiliary_budget`, `mpsae_repair.py:159` | Giữ K của code phục vụ; chỉ giảm B của nhánh reconstruction raw |

CSRv2 giữ loss CSR; MPSAEv2 giữ loss MPSAE. V2 không tự thêm residual repair hoặc teacher neighborhood. Nhãn `mmrsae-annealed` không đồng nghĩa `mpsaev2`.

## 8. Các protocol và pipeline đã cài đặt

### 8.1 MMRSAE ImageNet hiện tại

[mmrsae_imagenet.py](mmrsae_imagenet.py), 603 dòng, dùng protocol `mmrsae-imagenet-manual-v1`. Mỗi lần gọi Python fit một variant, một K, một seed trên các backbone được chọn; shell có thể gọi nhiều variant tuần tự.

| Khối | Vị trí | Chức năng |
|---|---|---|
| Parse/validate | `build_parser:38`, `validate_args:137` | Cấu hình phương pháp, dữ liệu, GPU, workload |
| Model config | `method_config:193` | Tạo `MMRSAEConfig` theo input dimension |
| Evaluate | `evaluate:266` | Gọi benchmark chung cho code sparse hoặc dense parent |
| Diagnostics | `diagnostic:291` | Kiểm tra objective/gradient/support tại các mốc |
| Fit/resume | `fit_and_evaluate:349` | Train, landmark refresh, checkpoint, đánh giá và ghi summary |
| Điều phối | `main:504` | Plan hoặc GPU preflight → chuẩn bị dữ liệu → fit → báo cáo |

Mặc định dev dùng 128.000 hàng fit/gallery và 10.000 query lấy từ training, K=8, batch=1024, seed mô hình 1227, data seed 42, tối đa 500 update mỗi fit, epoch limit 10. Model/optimizer dùng float32, AMP và TF32 tắt. Adam: lr=4e-5, weight decay=1e-4, eps=6,25e-10. Landmark refresh và checkpoint mặc định mỗi 25 update.

Checkpoint `last.pt` gồm model, Adam state theo tên parameter, RNG, mask RNG, bank, vị trí update và thông tin alpha. Recipe, code và data identity được đối chiếu khi resume. Các file chính gồm `fit_identity.json`, `training.jsonl`, diagnostics, retrieval metrics/predictions, `summary.json`, `failure.json` nếu lỗi.

**Cấu hình mẫu đang chọn sáu phương pháp.** [.env.example](.env.example), dòng 38, có:

```dotenv
MMRSAE_VARIANTS=mmrsae,mmrsae-rel,mmrsae-wide,mmrsae-annealed,mpsae,csr
```

`run_mmrsae_imagenet.sh:112` ưu tiên danh sách này trước `MMRSAE_VARIANT`; dòng 130 trở đi lần lượt gọi Python cho từng phương pháp. Với hai backbone và cap 500 update, workload cấu hình này là **12 fit, tối đa 6.000 optimizer update** nếu chạy đầy đủ, không phải chỉ hai fit/1.000 update. Đây là phép suy ra từ template; không phải xác nhận cấu hình riêng trong `.env` hoặc một job đã chạy. Python mặc định riêng lẻ vẫn chọn một variant.

Compose luôn đặt `EXPERIMENT_SCOPE=mmrsae` trong [compose.yaml](compose.yaml), dòng 14. Những trường historical/transfer trong `.env.example` không khiến service Compose này tự chuyển sang suite khác.

### 8.2 Các thí nghiệm CSR/MPSAE ImageNet cũ

[csr_vs_mmpot_imagenet.py](csr_vs_mmpot_imagenet.py), 5.145 dòng, chứa model, solver, train, retrieval và reporting trong cùng file.

| Protocol | Dữ liệu và baseline | Cách fit chính |
|---|---|---|
| `paper_ff2048` | Released FF2048 ResNet-50, d=2048; đường MRL/CSR công bố lưu làm reference | CSR và MPSAE độc lập cho từng K trong 8,16,32,64,128,256,512,1024,2048 |
| `paper_rn18_study` | `timm/resnet18.a1_in1k`, d=512; dense reference đo tại chỗ | K=8,16,64,256; ba seed 1227,1228,1229; báo mean, sample std và paired difference |
| `torchvision_ablation` | Frozen torchvision parent; có nhánh local Matryoshka training | Nghiên cứu ablation riêng, không được gộp thành paper reproduction |

FF2048 strict dùng h=8192, 10 epoch, batch=4096, gamma=0,1; RN18 dùng batch=1024, gamma=1,0, giữ h=8192. Các hyperparameter Adam và hệ số reconstruction theo bảng trên. Các nhánh v2 là exploratory extensions; strict paper comparison tập trung vào fixed-K v1.

Đường MRL công bố là reference hằng số, không phải local MRL được train trong paper mode. FF2048 có integrity gate so sánh dense full-input 1-NN với reference 71,19%, tolerance 0,05 điểm phần trăm. RN18 không có đường MRL công bố tương ứng được tự tạo thêm.

Runner này lưu history, bảng, plots, metrics và provenance; không lưu model/optimizer checkpoint như runner MMRSAE.

### 8.3 Repair study có đối chứng từng thành phần

[experiments.json](experiments.json), [run_matrix.csv](run_matrix.csv), [make_plan.py](make_plan.py) và [mpsae_repair_experiment.py](mpsae_repair_experiment.py) định nghĩa protocol `mpsae-repair-neighborhood-v1`.

| Arm | Nội dung |
|---|---|
| W | Warm model MPSAE nguyên bản, 250 update |
| C | Tiếp tục MPSAE, làm control; chữ C ở đây không có nghĩa CSR |
| R | Thêm repair, giữ angular OT, không mask/KL |
| M | Thêm mask vào R |
| O | Thêm KL vào M, giữ angular OT |
| S | Repair + mask + KL, tắt OT |
| N | Repair + mask + KL + mixed OT |
| F | N với auxiliary reconstruction wide |
| A | N với auxiliary reconstruction annealed |

Các continuation dùng cùng checkpoint W để đối chiếu. Triage gồm W/C/S/N, tổng 1.000 update; attribution R/M/O thêm 750; wide F/A thêm 500. Full pilot nếu đi hết các stage là 2.250 update. S trong pilot vẫn thừa hưởng lịch sử OT của W; fresh confirmation là phép kiểm tra riêng cho S từ đầu không OT.

Runner có các stage `check`, `warm`, `triage`, `attribution`, `wide`, `arm`, `timing`, `lock`, `holdout`, `confirm`. Nó có cheap screen, khóa candidate/checkpoint/code identity, holdout mở một lần và bootstrap paired query differences. Default gate đòi gain tối thiểu 0,3 điểm phần trăm, encoding ratio tối đa 3 và positive lower CI tại holdout. Confirmation train mới trên full data với hai seed, không dùng tiếp checkpoint pilot.

`experiments.json` để `source.revision` và `train_sha256` là null. Template này lập plan được; stage training yêu cầu điền trusted source pins. Đây là cơ chế có trong code, không phải xác nhận các stage đã chạy.

Các arm S/N/F/A có thành phần tương ứng với `mmrsae-rel`/`mmrsae`/`mmrsae-wide`/`mmrsae-annealed`, nhưng hai runner khác khởi tạo, branch history và protocol; không xem checkpoint/kết quả của chúng là tự động tương đương.

### 8.4 Frozen transfer

[transfer_experiments.py](transfer_experiments.py) điều phối CSR, MPSAE, CSRv2, MPSAEv2 và Matryoshka trên detection, mask, pose. Suite này chưa liệt kê MMRSAE trong danh sách methods.

- `transfer_representations.py:384`, `train_compressor`, import `TopKSAE`, NCL và matching gap từ runner ImageNet cũ. Train compressor có checkpoint/resume theo epoch; sparse budget được hỗ trợ là 8,16,64,256.
- `transfer_data.py` dựng crop/proposal, VOC2007, VOC2012 và PASCAL3D+; có chuẩn hóa rotation convention và cache manifest.
- `transfer_tasks.py` giữ indices qua `SparseBatch`, `FeatureStore` và `IndexedLinear`; dùng readout affine/MLP, đánh giá detection/mask/pose, controls và paired bootstrap.
- `transfer_reporting.py` vẽ kết quả transfer.

Compressor transfer dùng recipe riêng: batch=1024, gamma=1,0, `drop_last=True`, các v2 anneal từ `max(64,4K)` về endpoint K. Không mặc nhiên coi recipe này là strict FF2048 Table 3 hoặc cùng protocol MMRSAE.

### 8.5 Các runner dense cũ có OT

`matryoshka_mmpot_experiment.py` dùng `PairwiseScaleCost:601` và `GreenkhornPartialOT:631` cho proxy hai marginal giữa dense prefixes. `matryoshka_real_mmpot_experiment.py` dùng `ScaleCostBuilder:607` và `GreenkhornMMPOTSolver:668`, hỗ trợ hai hoặc ba marginal explicit.

Chúng train dense Matryoshka/classifier, có checkpoint, k-NN và linear probe riêng. Không phải cài đặt sparse MPSAE và không được default MMRSAE workflow gọi.

## 9. Dữ liệu, serving và đo hiệu năng

| File | Khối chính | Vai trò |
|---|---|---|
| [mmrsae_data.py](mmrsae_data.py) | `_select_pools:93`, `_precomputed_split:193`, `prepare_imagenet_features:355` | Local/Hub prepared NPZ+JSON, provenance, chia dev/official và landmarks |
| [repair_data.py](repair_data.py) | `inspect_npz:144`, `open_feature_cache:235`, `load_splits:299` | Đọc schema, xác minh hash, direct/streamed memory mapping, khóa cache |
| [prepare_splits.py](prepare_splits.py) | `build_indices:21`, `prepare_splits:68` | Các pool cân bằng, deterministic cho repair study |
| [export_rn18_embeddings.py](export_rn18_embeddings.py) | `export_split:270` | Export RN18 từ raw images cho legacy workflow |
| [download_imagenet.py](download_imagenet.py) | `validate_split:92`, `main:198` | Download/validate raw ImageNet theo cấu hình |
| [mmrsae_gpu.py](mmrsae_gpu.py) | `_retrieval_probe:34`, `preflight:130` | Kiểm tra môi trường và probe GPU retrieval trước chuẩn bị dữ liệu/train |
| [repair_benchmark.py](repair_benchmark.py) | `_encoded_cache:99`, `_resident_encoding_measurement:226`, `benchmark:276` | Encode/cache, timing encoding riêng, accuracy/search timing, predictions và memory accounting |
| [sparse_gpu_retrieval.py](sparse_gpu_retrieval.py) | `CudaSparseIndex:139`, `add:215`, `prepare:303`, `search:335` | GPU inverted index trên indices int32/values float32 |
| [csrc/sparse_retrieval.cpp](csrc/sparse_retrieval.cpp) | `sparse_accumulate`, pybind | Binding và kiểm tra input cho extension |
| [csrc/sparse_retrieval.cu](csrc/sparse_retrieval.cu) | `accumulate_postings_kernel:20` | Duyệt postings theo tọa độ query, atomic cộng tích sparse |
| [retrieval_timing.py](retrieval_timing.py) | `CudaSearchTimer:26` | Warmup, repeated CUDA-event timing và synchronized wall timing |
| [repair_runtime.py](repair_runtime.py) | `pack_adam:40`, `restore_adam:74`, `rng_state:148`, `save_checkpoint:216` | Adam state theo tên, RNG, deterministic batches và checkpoint |
| [wandb_tracking.py](wandb_tracking.py) | `init_wandb:71`, `log_wandb_metrics:158` | Theo dõi config, loss, gradient diagnostics, GPU và kết quả |

Nguồn mặc định MMRSAE là prepared NPZ/JSON trong `MMRSAE_EMBEDDINGS`, hoặc tải từng cặp thiếu từ repository cấu hình. Local pairs được ưu tiên. Dev chỉ cần train archive; official mở thêm validation. RN18 uncompressed NPZ được map trực tiếp; compressed FF2048 được stream sang NPY disk map tái sử dụng. Dữ liệu được chuyển sang float32 theo batch, không đọc toàn feature matrix vào RAM.

Dev query tách khỏi fit và anchors. Với subset mặc định, fit cũng tách khỏi anchors; khi lấy full eligible training pool, anchors có thể thuộc pool fit nên profile exclusion theo minibatch ID vẫn cần thiết.

Retrieval exact 1-NN dùng L2-normalized embeddings. Dense parent dùng GPU FAISS; sparse code dùng CUDA inverted index, tận dụng dot product tương đương ranking theo squared L2 cho vector unit-norm hợp lệ. Gallery phải resident trên GPU; chunking giới hạn workspace. Hệ thống không tự fallback sang CPU.

Search timing mặc định query batch=32, warmup=3, repeats=5. Compile extension, encode, transfer và dựng index được tách khỏi search timing; benchmark hiện tại còn đo encoding riêng. Thời gian mỗi query là amortized theo batch, không đồng nghĩa latency một query độc lập.

MMRSAE mặc định cho phép code toàn 0, ghi nhận tần suất, và tính empty query là retrieval miss. Helper lõi và đường legacy có tùy chọn/kiểm tra nghiêm hơn. Phép đổi cosine→L2 được giải thích cho vector unit-norm hợp lệ; code zero là trường hợp riêng cần đọc cùng chính sách này.

Code sparse không tự bảo đảm nhanh hơn dense: postings phổ biến, score buffers, reduction và K lớn đều ảnh hưởng chi phí. Atomic float32 cũng có thể thay đổi các thứ hạng gần hòa. Equal K không đồng nghĩa equal bytes: payload chỉ riêng int32 index + float32 value đã xấp xỉ 8K byte, chưa tính metadata, so với 4K byte của dense prefix float32.

## 10. Diagnostics và kiểm thử có trong code

`reconstruction_terms` ghi actual nnz, zero-code fraction, tỷ lệ dưới K, preactivation thứ K không dương, tail energy 2K/4K, bias và dead fraction. `LandmarkBank` ghi overlap, entropy, neighbor preservation và key drift. `mixed_matching_gap` ghi cost ranges, std/eta, mass error, capacity violation, solver iterations và diagonal/shuffled affinity.

`gradient_diagnostics`, `mpsae_repair.py:810`, phân tách gradient reconstruction/KL/angular OT/neighborhood OT và đếm selected coefficients có gradient gần 0. Các phép này giúp nhận biết support shrinkage hoặc thiếu tín hiệu học.

Ví dụ giới hạn đã được kiểm tra trong source tests: khi sparse query và bank không chồng support, gradient similarity có thể biến mất; với K=1, chuẩn hóa một hệ số dương có thể làm similarity không phụ thuộc độ lớn của hệ số. Repair, masking và auxiliary budget là các giả thuyết xử lý khó khăn đó, chưa phải bằng chứng thành công thực nghiệm.

| File kiểm thử | Method `test_*` | Phạm vi |
|---|---:|---|
| [check_reference.py](check_reference.py) | 16 | Baseline equivalence, deterministic Top-K, IHT, RNG, mask, indexed gradient, bank, OT envelope gradient và schedule |
| [tests/test_embedding_data.py](tests/test_embedding_data.py) | 15 | ZIP64, Fortran/byte order, compressed map, local/Hub routing, hash, dev/official boundaries |
| [tests/test_neighborhood_cost.py](tests/test_neighborhood_cost.py) | 15 | Công thức affinity, nonnegative range, precision, masking/log offsets, gradients, defaults rho |
| [tests/test_ot_cost_safety.py](tests/test_ot_cost_safety.py) | 11 | Invalid costs/settings, feasibility, negative-gap refinement và envelope gradient |
| [tests/test_repair_data.py](tests/test_repair_data.py) | 16 | Planner, nguồn dữ liệu, split/cache, locks và query bootstrap |
| [tests/test_repair_runner.py](tests/test_repair_runner.py) | 10 | Screen/gate, immutable configuration, candidate lock và holdout mở một lần |
| [tests/test_repair_runtime.py](tests/test_repair_runtime.py) | 4 | Adam theo tên, branch moments, independent RNG và resume |

87 là số method được khai báo, không phải 87 test được chạy/pass trong lần rà soát này. [VALIDATION.md](VALIDATION.md) lưu lịch sử checks trước đó; lịch sử đó không thay thế việc kiểm tra toàn bộ trạng thái hiện tại hoặc đo ImageNet/CUDA.

## 11. Bản đồ toàn bộ 31 file Python ở gốc

| File | Dòng | Trách nhiệm |
|---|---:|---|
| `mmrsae.py` | 344 | Public API, config, presets và CSR baseline |
| `mpsae_repair.py` | 826 | SAE, Top-K, repair, masking, bank, KL, OT và diagnostics |
| `csr_vs_mmpot_imagenet.py` | 5.145 | Historical CSR/MPSAE v1–v2 ImageNet và paper protocols |
| `mmrsae_imagenet.py` | 603 | Manual MMRSAE training/evaluation runner |
| `mpsae_repair_experiment.py` | 971 | Staged research runner và quyết định screen/gate |
| `make_plan.py` | 311 | Validate research config, compute budget và export matrix |
| `mmrsae_data.py` | 439 | Adapter prepared embeddings và pools MMRSAE |
| `repair_data.py` | 329 | Archive/schema/hash/cache/locks |
| `prepare_splits.py` | 181 | Deterministic research splits |
| `export_rn18_embeddings.py` | 335 | RN18 feature export |
| `download_imagenet.py` | 279 | Raw ImageNet data preparation |
| `mmrsae_gpu.py` | 222 | GPU preflight và retrieval probe |
| `repair_benchmark.py` | 521 | Compact encoding cache và benchmark |
| `sparse_gpu_retrieval.py` | 382 | Sparse CUDA inverted index |
| `retrieval_timing.py` | 130 | Search timer chung |
| `repair_runtime.py` | 267 | Reproducibility, optimizer state và checkpoint |
| `wandb_tracking.py` | 195 | W&B integration |
| `mmrsae_reporting.py` | 353 | Thu thập run, bảng và plots MMRSAE |
| `aggregate_backbone_results.py` | 1.629 | Aggregate legacy ImageNet, validate protocols, bảng paper, plots và ZIP |
| `aggregate_all_experiments.py` | 174 | Bundle/tổng hợp các experiment family cũ |
| `experiment_reporting.py` | 365 | Reporting cho các thí nghiệm dense Matryoshka cũ |
| `summarize_results.py` | 198 | Đọc summary của dense MRL/MMPOT cũ |
| `compare_predictions.py` | 114 | Paired prediction comparison và uncertainty |
| `transfer_experiments.py` | 540 | Frozen transfer orchestration, selection và packaging |
| `transfer_representations.py` | 532 | Parent models, feature extraction, compressor fit/resume và sparse cache |
| `transfer_data.py` | 894 | Downstream datasets, crop/proposal/mask/pose records |
| `transfer_tasks.py` | 1.555 | Sparse-preserving readouts, task training, metrics, controls và bootstrap |
| `transfer_reporting.py` | 139 | Transfer figures |
| `matryoshka_mmpot_experiment.py` | 1.578 | Dense MRL và two-marginal OT proxy |
| `matryoshka_real_mmpot_experiment.py` | 1.794 | Dense MRL và explicit multimarginal OT |
| `check_reference.py` | 316 | Mathematical/reference checks |

`mmrsae_reporting.py` là đường báo cáo hiện tại; `summarize_results.py` không phải model hay loss implementation của MMRSAE/MPSAE.

## 12. Launchers, cấu hình và tài liệu

| File | Vai trò |
|---|---|
| `run_mmrsae_imagenet.sh` | Chuyển env thành CLI, chạy các method tuần tự và gọi reporting |
| `run_mpsae_repair.sh` | Chạy một stage repair research |
| `run_imagenet.sh` | Entry point tiện dụng cho repair/MMRSAE, init/plan/preflight/prepare/run/logs |
| `run_all_experiments_docker.sh` | Host Docker launcher, chọn scope và persistent mounts |
| `run_all_experiments_container.sh` | Điều phối trong container theo scope |
| `run_full_pipeline.sh` | Alias host launcher |
| `run_csr_vs_mmpot_imagenet.sh` | Entry point historical CSR/MPSAE ImageNet |
| `run_transfer_experiments.sh` | Entry point frozen transfer |
| `run_imagenet_download.sh` | Raw ImageNet download/validation |
| `experiment_shell_common.sh` | Python/env/config helpers cho shell |
| `container_pipeline.sh` | Pipeline cho các thí nghiệm dense cũ |
| `compose.yaml`, `Dockerfile.all-experiments` | Container/GPU/data/output setup |
| `.env.example`, `mmrsae.env.example` | Cấu hình mẫu MMRSAE và các scope lịch sử |
| `repair.env.example`, `experiments.json`, `run_matrix.csv` | Recipe và arm matrix repair research |
| `rn18_study.env.example` | Recipe riêng nghiên cứu RN18 Appendix E.1 |
| `requirements.txt` | Dependencies |

Tài liệu đọc theo mục đích:

- [MMRSAE_IMAGENET.md](MMRSAE_IMAGENET.md): current API, data và manual workflow.
- [MPSAE_NOVELTY_VS_MRL_CSR.md](MPSAE_NOVELTY_VS_MRL_CSR.md): giải thích khác biệt phương pháp.
- [full_theory_detail.md](full_theory_detail.md): lý thuyết repair/neighborhood/auxiliary annealing.
- [full_experiment_detail.md](full_experiment_detail.md): staged experimental design.
- [CSR_VS_MMPOT_IMAGENET_EXPERIMENT.md](CSR_VS_MMPOT_IMAGENET_EXPERIMENT.md): paper protocols cũ.
- [EXPERIMENT_DETAILS_AND_BENCHMARKS.md](EXPERIMENT_DETAILS_AND_BENCHMARKS.md): broader experiment/transfer reference.
- [REMARK.md](REMARK.md): scientific audit và giới hạn so sánh.
- [VALIDATION.md](VALIDATION.md): lịch sử kiểm tra đã thực hiện.
- [README.md](README.md): cách vận hành tổng quát.

Các ý tưởng trong tài liệu được đối chiếu với code khi thống kê. Cần đọc tên method cùng protocol, parent, dữ liệu, budget, seed và recipe; sự có mặt của model/loss/test trong repository chưa xác lập accuracy, tốc độ hay lợi ích của từng thành phần. Trong thư mục gốc được rà soát không có thư mục run output để dùng làm bằng chứng thực nghiệm cho các so sánh đó.
