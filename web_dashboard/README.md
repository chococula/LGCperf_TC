# LGCperf 결과 대시보드 (Google Apps Script)

측정 PC가 Run마다 결과를 Google 시트에 올리고, 같은 Apps Script가 그 시트를 읽어 웹 대시보드로 보여줍니다.
데이터는 Google Drive의 스프레드시트에 TC별 탭으로 쌓입니다.

```
측정 PC (Omedia_TC01.py) ──POST──▶ Apps Script 웹앱 ──▶ Google 시트 (TC별 탭)
                                          │
브라우저 ◀──────── 대시보드 (Index.html) ◀──┘
```

## 1회 설정

1. Google Drive에서 새 스프레드시트를 만듭니다 (이름 예: `LGCperf Results`).
2. 시트 메뉴 **확장 프로그램 → Apps Script**를 엽니다.
3. 기본 `Code.gs` 내용을 이 폴더의 [Code.gs](Code.gs)로 바꿉니다.
   - `UPLOAD_TOKEN`을 임의의 긴 문자열로 바꿉니다 (측정 PC 설정과 같아야 함).
4. **배포 → 새 배포 → 유형: 웹 앱**
   - 실행 사용자: **나**
   - 액세스 권한: **모든 사용자** (측정 PC가 로그인 없이 POST해야 하므로)
   - 권한 승인 후 나오는 **웹 앱 URL**(`https://script.google.com/macros/s/.../exec`)을 복사합니다.
5. 측정 PC의 `LGCperf_TC` 폴더에서 `result_upload_config.example.json`을 `result_upload_config.json`으로 복사하고
   `url`에 웹 앱 URL, `token`에 3번의 토큰을 넣습니다.
6. `gh-pages` 브랜치의 `config.js`에 웹 앱 URL을 넣어 push합니다 (대시보드·에디터가 실데이터를 읽음). (이 파일은 git에 올라가지 않습니다.)

## 사용

- 측정 스크립트를 평소대로 실행하면 Run마다 결과가 자동 업로드됩니다. 업로드는 백그라운드라 측정 타이밍에 영향이 없습니다.
- 네트워크 오류로 실패한 행은 `result_upload_pending.jsonl`에 남았다가 다음 업로드 때 재전송됩니다.
- `result_upload_config.json`이 없으면 업로드는 건너뛰고 로컬 CSV만 저장합니다.
- 대시보드: https://chococula.github.io/LGCperf_TC/ — 30초마다 자동 새로고침되고 새 Run은 잠깐 하이라이트됩니다.
- 에디터: https://chococula.github.io/LGCperf_TC/editor.html

## 공식 페이지 (GitHub Pages)

- 주소: https://chococula.github.io/LGCperf_TC/
- `gh-pages` 브랜치의 `index.html`(= 이 폴더의 `Index.html`)과 `config.js`로 서비스됩니다.
- Apps Script를 배포한 뒤 `gh-pages` 브랜치의 `config.js`에 있는 `LGCPERF_API_URL`을 웹 앱 URL로 바꿔 push하면 실데이터가 표시됩니다.
  URL이 placeholder(`XXXXXXXX`)인 동안에는 샘플 데이터가 보입니다.

## 데이터 관리

- **기간 선택**: From/To 날짜, Today / 7 days / 30 days / All, Session(측정 스크립트 1회 실행 단위) 필터.
- **잘못된 데이터 제외**: 우측 상단에 관리자 키(`Code.gs`의 `ADMIN_KEY`)를 입력해 Unlock하면 각 행에 `Exclude…` 버튼이 생깁니다.
  사유를 입력하면 시트의 `Excluded=Y`, `Exclude_Reason`, `Excluded_At`이 기록됩니다. 행은 삭제되지 않으며 `Restore`로 되돌릴 수 있습니다.
  시트에서 직접 `Excluded` 칸에 `Y`를 넣어도 같습니다.
- **통계 기준**: 제외되지 않은 PASS Run만 평균/중앙값/최소/최대/표준편차와 그래프에 사용합니다.
- **Outlier? 표시**: 같은 빌드(SoC/SWV/LGCV) 안에서 중앙값보다 15% 넘게 벗어나고 robust z > 3.5인 값(빌드당 5개 이상일 때)에 붙는 힌트입니다. 자동 제외하지 않으니 원인을 확인한 뒤 제외하세요.

## 데이터 에디터 (editor.html)

- **셀 편집**: 셀 클릭 → 수정 → Enter. 수정한 셀은 노랗게 표시되고 **Save changes**로 한 번에 반영됩니다 (Discard로 취소).
- **선택 행 처리**: 체크박스로 고른 뒤 **Exclude…**(사유 입력, 통계 제외) / **Restore** / **Delete…**(영구 삭제, 확인 필요).
  Google 시트에서 삭제한 행은 `_Deleted` 탭에 먼저 복사됩니다.
- **로컬 CSV**: **Open CSV file…**로 `Omedia_TC01_response.csv`를 열어 같은 방식으로 편집하고 **Download edited CSV**로 저장합니다.
  (시트가 없어도 사용 가능, 관리자 키 불필요)
- Google 시트 편집은 관리자 키로 Unlock해야 합니다. 행은 `Row_ID`로 식별하므로 시트에서 행을 지우거나 정렬해도 엉뚱한 행이 수정되지 않습니다.

## 코드 수정 후

Apps Script에서 코드를 고친 뒤에는 **배포 → 배포 관리 → 편집 → 버전: 새 버전**으로 다시 배포해야 반영됩니다 (URL은 유지).

## 다른 TC에 붙이기

```python
from result_uploader import upload_result, wait_for_uploads

upload_result({'TC': 'LGC_TC1_RegularBoot', 'Run': 1, 'Timestamp': ts,
               'SoC': soc, 'SWV': swv, 'LGCV': lgcv, 'Boot_Time_ms': 12345.6, 'Status': 'PASS'})
...
wait_for_uploads()  # 종료 전에 호출
```

`TC` 값마다 시트 탭이 자동 생성되고, `_ms`로 끝나는 컬럼은 대시보드에서 지표로 선택할 수 있습니다.
