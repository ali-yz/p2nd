| Algorithm         | Linkage / Method | Min Cluster Size (MCS) | Min Samples (MS) | Distance Threshold | AMI_core | DBCV_core | Notes                    |
| ----------------- | ---------------- | ---------------------: | ---------------: | -----------------: | -------: | --------: | ------------------------ |
| **HDBSCAN**       | eom              |                     25 |             none |                  – | 0.129997 | -0.441329 | baseline run             |
| HDBSCAN           | eom              |                     25 |                5 |                  – | 0.125884 | -0.368776 |                          |
| HDBSCAN           | eom              |                     25 |               10 |                  – | 0.128610 | -0.394783 |                          |
| HDBSCAN           | eom              |                     25 |               25 |                  – | 0.129997 | -0.441329 | same as MS=none          |
| HDBSCAN           | eom              |                     25 |               50 |                  – | 0.129891 | -0.439140 |                          |
| HDBSCAN           | eom              |                     25 |              100 |                  – | 0.129426 | -0.517427 |                          |
| HDBSCAN           | eom              |                     50 |             none |                  – | 0.130028 | -0.447852 |                          |
| HDBSCAN           | eom              |                     50 |                5 |                  – | 0.130212 | -0.315078 | best DBCV among MCS=50   |
| HDBSCAN           | eom              |                     50 |               10 |                  – | 0.129731 | -0.387322 |                          |
| HDBSCAN           | eom              |                     50 |               25 |                  – | 0.129938 | -0.444762 |                          |
| HDBSCAN           | eom              |                     50 |               50 |                  – | 0.130028 | -0.447852 |                          |
| HDBSCAN           | eom              |                     50 |              100 |                  – | 0.129380 | -0.526167 |                          |
| HDBSCAN           | eom              |                    100 |             none |                  – | 0.131154 | -0.546782 |                          |
| HDBSCAN           | eom              |                    100 |                5 |                  – | 0.129191 | -0.340448 |                          |
| HDBSCAN           | eom              |                    100 |               10 |                  – | 0.129182 | -0.360582 |                          |
| HDBSCAN           | eom              |                    100 |               25 |                  – | 0.128816 | -0.431265 |                          |
| HDBSCAN           | eom              |                    100 |               50 |                  – | 0.130212 | -0.512254 |                          |
| HDBSCAN           | eom              |                    100 |              100 |                  – | 0.131154 | -0.546782 | same as MS=none          |
| HDBSCAN           | eom              |                    200 |             none |                  – | 0.132668 | -0.573032 | highest AMI overall      |
| HDBSCAN           | eom              |                    200 |                5 |                  – | 0.129564 | -0.104768 | best DBCV overall        |
| HDBSCAN           | eom              |                    200 |               10 |                  – | 0.129923 | -0.302295 |                          |
| HDBSCAN           | eom              |                    200 |               25 |                  – | 0.130285 | -0.397754 |                          |
| HDBSCAN           | eom              |                    200 |               50 |                  – | 0.130462 | -0.475892 |                          |
| HDBSCAN           | eom              |                    200 |              100 |                  – | 0.132024 | -0.524115 |                          |
| HDBSCAN           | eom              |                    400 |             none |                  – | 0.131130 | -0.696673 |                          |
| HDBSCAN           | eom              |                    400 |                5 |                  – | 0.130229 | -0.115233 |                          |
| HDBSCAN           | eom              |                    400 |               10 |                  – | 0.130279 | -0.312876 |                          |
| HDBSCAN           | eom              |                    400 |               25 |                  – | 0.130604 | -0.401603 |                          |
| HDBSCAN           | eom              |                    400 |               50 |                  – | 0.130824 | -0.476772 |                          |
| HDBSCAN           | eom              |                    400 |              100 |                  – | 0.129643 | -0.535985 |                          |
| **Agglomerative** | ward             |                      – |                – |                 10 | 0.115180 | -0.755236 |                          |
| Agglomerative     | ward             |                      – |                – |                 15 | 0.120133 | -0.799524 |                          |
| Agglomerative     | ward             |                      – |                – |                 20 | 0.123577 | -0.814391 |                          |
| Agglomerative     | ward             |                      – |                – |                 25 | 0.125797 | -0.820459 |                          |
| Agglomerative     | ward             |                      – |                – |                 30 | 0.129413 | -0.812790 |                          |
| Agglomerative     | ward             |                      – |                – |                 35 | 0.129816 | -0.822075 |                          |
| Agglomerative     | ward             |                      – |                – |                 40 | 0.131655 | -0.814854 |                          |
| Agglomerative     | ward             |                      – |                – |                 50 | 0.132576 | -0.820050 | highest AMI for agglom   |
| Agglomerative     | ward             |                      – |                – |                 60 | 0.132630 | -0.799244 |                          |
| Agglomerative     | ward             |                      – |                – |                 80 | 0.130236 | -0.708781 |                          |
| Agglomerative     | ward             |                      – |                – |                100 | 0.131359 | -0.711700 |                          |
| Agglomerative     | complete         |                      – |                – |                 10 |        – |         – | **Process killed (OOM)** |
| Agglomerative     | average          |                      – |                – |                 30 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | average          |                      – |                – |                 40 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | average          |                      – |                – |                 50 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | average          |                      – |                – |                 60 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | average          |                      – |                – |                 80 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | single           |                      – |                – |                 30 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | single           |                      – |                – |                 40 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | single           |                      – |                – |                 50 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | single           |                      – |                – |                 60 |        – |         – | **Exit 137 (killed)**    |
| Agglomerative     | single           |                      – |                – |                 80 |        – |         – | **Exit 137 (killed)**    |
