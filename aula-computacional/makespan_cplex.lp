\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: makespan

Minimize
 obj: y
Subject To
 c1: 13 x_0_0 + 16 x_0_1 + 3 x_0_2 + 24 x_0_3 + 11 x_0_4 + 24 x_0_5 + 7 x_0_6
     + 18 x_0_7 + 12 x_0_8 + 9 x_0_9 + 7 x_0_10 + 14 x_0_11 + 12 x_0_12
     + 15 x_0_13 + 4 x_0_14 + 13 x_0_15 + 3 x_0_16 + 10 x_0_17 + x_0_18
     + 4 x_0_19 - y <= 0
 c2: 23 x_1_0 + x_1_1 + 23 x_1_2 + 18 x_1_3 + 7 x_1_4 + 18 x_1_5 + 15 x_1_6
     + 2 x_1_7 + 17 x_1_8 + 5 x_1_9 + 18 x_1_10 + 17 x_1_11 + 13 x_1_12 + x_1_13
     + 24 x_1_14 + 10 x_1_15 + 12 x_1_16 + 23 x_1_17 + 6 x_1_18 + 8 x_1_19 - y
      <= 0
 c3: 4 x_2_0 + 19 x_2_1 + 10 x_2_2 + 7 x_2_3 + 4 x_2_4 + 24 x_2_5 + 19 x_2_6
     + 16 x_2_7 + 9 x_2_8 + 3 x_2_9 + 17 x_2_10 + 12 x_2_11 + 19 x_2_12 + x_2_13
     + 4 x_2_14 + 8 x_2_15 + 16 x_2_16 + 21 x_2_17 + 3 x_2_18 + 21 x_2_19 - y
      <= 0
 c4: 4 x_3_0 + x_3_1 + 24 x_3_2 + x_3_3 + 8 x_3_4 + 2 x_3_5 + 24 x_3_6 + 5 x_3_7
     + 10 x_3_8 + 8 x_3_9 + 21 x_3_10 + 9 x_3_11 + 17 x_3_12 + 19 x_3_13
     + 7 x_3_14 + 13 x_3_15 + 16 x_3_16 + x_3_17 + 2 x_3_18 + 23 x_3_19 - y <= 
     0
 c5: 24 x_4_0 + 19 x_4_1 + 6 x_4_2 + 4 x_4_3 + 21 x_4_4 + 24 x_4_5 + 9 x_4_6
     + 21 x_4_7 + x_4_8 + 7 x_4_9 + 15 x_4_10 + 18 x_4_11 + 15 x_4_12
     + 19 x_4_13 + 7 x_4_14 + 24 x_4_15 + 20 x_4_16 + 5 x_4_17 + 16 x_4_18
     + 13 x_4_19 - y <= 0
 c6: x_0_0 + x_1_0 + x_2_0 + x_3_0 + x_4_0 = 1
 c7: x_0_1 + x_1_1 + x_2_1 + x_3_1 + x_4_1 = 1
 c8: x_0_2 + x_1_2 + x_2_2 + x_3_2 + x_4_2 = 1
 c9: x_0_3 + x_1_3 + x_2_3 + x_3_3 + x_4_3 = 1
 c10: x_0_4 + x_1_4 + x_2_4 + x_3_4 + x_4_4 = 1
 c11: x_0_5 + x_1_5 + x_2_5 + x_3_5 + x_4_5 = 1
 c12: x_0_6 + x_1_6 + x_2_6 + x_3_6 + x_4_6 = 1
 c13: x_0_7 + x_1_7 + x_2_7 + x_3_7 + x_4_7 = 1
 c14: x_0_8 + x_1_8 + x_2_8 + x_3_8 + x_4_8 = 1
 c15: x_0_9 + x_1_9 + x_2_9 + x_3_9 + x_4_9 = 1
 c16: x_0_10 + x_1_10 + x_2_10 + x_3_10 + x_4_10 = 1
 c17: x_0_11 + x_1_11 + x_2_11 + x_3_11 + x_4_11 = 1
 c18: x_0_12 + x_1_12 + x_2_12 + x_3_12 + x_4_12 = 1
 c19: x_0_13 + x_1_13 + x_2_13 + x_3_13 + x_4_13 = 1
 c20: x_0_14 + x_1_14 + x_2_14 + x_3_14 + x_4_14 = 1
 c21: x_0_15 + x_1_15 + x_2_15 + x_3_15 + x_4_15 = 1
 c22: x_0_16 + x_1_16 + x_2_16 + x_3_16 + x_4_16 = 1
 c23: x_0_17 + x_1_17 + x_2_17 + x_3_17 + x_4_17 = 1
 c24: x_0_18 + x_1_18 + x_2_18 + x_3_18 + x_4_18 = 1
 c25: x_0_19 + x_1_19 + x_2_19 + x_3_19 + x_4_19 = 1

Bounds
 0 <= x_0_0 <= 1
 0 <= x_0_1 <= 1
 0 <= x_0_2 <= 1
 0 <= x_0_3 <= 1
 0 <= x_0_4 <= 1
 0 <= x_0_5 <= 1
 0 <= x_0_6 <= 1
 0 <= x_0_7 <= 1
 0 <= x_0_8 <= 1
 0 <= x_0_9 <= 1
 0 <= x_0_10 <= 1
 0 <= x_0_11 <= 1
 0 <= x_0_12 <= 1
 0 <= x_0_13 <= 1
 0 <= x_0_14 <= 1
 0 <= x_0_15 <= 1
 0 <= x_0_16 <= 1
 0 <= x_0_17 <= 1
 0 <= x_0_18 <= 1
 0 <= x_0_19 <= 1
 0 <= x_1_0 <= 1
 0 <= x_1_1 <= 1
 0 <= x_1_2 <= 1
 0 <= x_1_3 <= 1
 0 <= x_1_4 <= 1
 0 <= x_1_5 <= 1
 0 <= x_1_6 <= 1
 0 <= x_1_7 <= 1
 0 <= x_1_8 <= 1
 0 <= x_1_9 <= 1
 0 <= x_1_10 <= 1
 0 <= x_1_11 <= 1
 0 <= x_1_12 <= 1
 0 <= x_1_13 <= 1
 0 <= x_1_14 <= 1
 0 <= x_1_15 <= 1
 0 <= x_1_16 <= 1
 0 <= x_1_17 <= 1
 0 <= x_1_18 <= 1
 0 <= x_1_19 <= 1
 0 <= x_2_0 <= 1
 0 <= x_2_1 <= 1
 0 <= x_2_2 <= 1
 0 <= x_2_3 <= 1
 0 <= x_2_4 <= 1
 0 <= x_2_5 <= 1
 0 <= x_2_6 <= 1
 0 <= x_2_7 <= 1
 0 <= x_2_8 <= 1
 0 <= x_2_9 <= 1
 0 <= x_2_10 <= 1
 0 <= x_2_11 <= 1
 0 <= x_2_12 <= 1
 0 <= x_2_13 <= 1
 0 <= x_2_14 <= 1
 0 <= x_2_15 <= 1
 0 <= x_2_16 <= 1
 0 <= x_2_17 <= 1
 0 <= x_2_18 <= 1
 0 <= x_2_19 <= 1
 0 <= x_3_0 <= 1
 0 <= x_3_1 <= 1
 0 <= x_3_2 <= 1
 0 <= x_3_3 <= 1
 0 <= x_3_4 <= 1
 0 <= x_3_5 <= 1
 0 <= x_3_6 <= 1
 0 <= x_3_7 <= 1
 0 <= x_3_8 <= 1
 0 <= x_3_9 <= 1
 0 <= x_3_10 <= 1
 0 <= x_3_11 <= 1
 0 <= x_3_12 <= 1
 0 <= x_3_13 <= 1
 0 <= x_3_14 <= 1
 0 <= x_3_15 <= 1
 0 <= x_3_16 <= 1
 0 <= x_3_17 <= 1
 0 <= x_3_18 <= 1
 0 <= x_3_19 <= 1
 0 <= x_4_0 <= 1
 0 <= x_4_1 <= 1
 0 <= x_4_2 <= 1
 0 <= x_4_3 <= 1
 0 <= x_4_4 <= 1
 0 <= x_4_5 <= 1
 0 <= x_4_6 <= 1
 0 <= x_4_7 <= 1
 0 <= x_4_8 <= 1
 0 <= x_4_9 <= 1
 0 <= x_4_10 <= 1
 0 <= x_4_11 <= 1
 0 <= x_4_12 <= 1
 0 <= x_4_13 <= 1
 0 <= x_4_14 <= 1
 0 <= x_4_15 <= 1
 0 <= x_4_16 <= 1
 0 <= x_4_17 <= 1
 0 <= x_4_18 <= 1
 0 <= x_4_19 <= 1

Binaries
 x_0_0 x_0_1 x_0_2 x_0_3 x_0_4 x_0_5 x_0_6 x_0_7 x_0_8 x_0_9 x_0_10 x_0_11
 x_0_12 x_0_13 x_0_14 x_0_15 x_0_16 x_0_17 x_0_18 x_0_19 x_1_0 x_1_1 x_1_2 x_1_3
 x_1_4 x_1_5 x_1_6 x_1_7 x_1_8 x_1_9 x_1_10 x_1_11 x_1_12 x_1_13 x_1_14 x_1_15
 x_1_16 x_1_17 x_1_18 x_1_19 x_2_0 x_2_1 x_2_2 x_2_3 x_2_4 x_2_5 x_2_6 x_2_7
 x_2_8 x_2_9 x_2_10 x_2_11 x_2_12 x_2_13 x_2_14 x_2_15 x_2_16 x_2_17 x_2_18
 x_2_19 x_3_0 x_3_1 x_3_2 x_3_3 x_3_4 x_3_5 x_3_6 x_3_7 x_3_8 x_3_9 x_3_10
 x_3_11 x_3_12 x_3_13 x_3_14 x_3_15 x_3_16 x_3_17 x_3_18 x_3_19 x_4_0 x_4_1
 x_4_2 x_4_3 x_4_4 x_4_5 x_4_6 x_4_7 x_4_8 x_4_9 x_4_10 x_4_11 x_4_12 x_4_13
 x_4_14 x_4_15 x_4_16 x_4_17 x_4_18 x_4_19

Generals
 y
End
