#include <iostream>
#include <cstring>

extern "C" {
    // 0:Red(U), 1:Green(L), 2:White(F), 3:Blue(R), 4:Orange(D), 5:Yellow(B)
    int state[54] = {
        0,0,0,0,0,0,0,0,0, // U (Red)
        1,1,1,1,1,1,1,1,1, // L (Green)
        2,2,2,2,2,2,2,2,2, // F (White)
        3,3,3,3,3,3,3,3,3, // R (Blue)
        4,4,4,4,4,4,4,4,4, // D (Orange)
        5,5,5,5,5,5,5,5,5  // B (Yellow)
    };

    const int solved_state[54] = {
        0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,5
    };

    bool is_solved() {
        for (int i = 0; i < 54; i++) {
            if (state[i] != solved_state[i]) return false;
        }
        return true;
    }

    void rotate_face_cw(int face) {
        int b = face * 9;
        int t[9];
        for(int i = 0; i < 9; i++) t[i] = state[b + i];
        state[b + 0] = t[6]; state[b + 1] = t[3]; state[b + 2] = t[0];
        state[b + 3] = t[7]; state[b + 4] = t[4]; state[b + 5] = t[1];
        state[b + 6] = t[8]; state[b + 7] = t[5]; state[b + 8] = t[2];
    }

    void move_U() {
        rotate_face_cw(0);
        int t[3] = {state[9], state[10], state[11]};
        state[9]  = state[18]; state[10] = state[19]; state[11] = state[20];
        state[18] = state[27]; state[19] = state[28]; state[20] = state[29];
        state[27] = state[45]; state[28] = state[46]; state[29] = state[47];
        state[45] = t[0];      state[46] = t[1];      state[47] = t[2];
    }

    void move_D() {
        rotate_face_cw(4);
        int t[3] = {state[24], state[25], state[26]};
        state[24] = state[15]; state[25] = state[16]; state[26] = state[17];
        state[15] = state[51]; state[16] = state[52]; state[17] = state[53];
        state[51] = state[33]; state[52] = state[34]; state[53] = state[35];
        state[33] = t[0];      state[34] = t[1];      state[35] = t[2];
    }

    void move_F() {
        rotate_face_cw(2);
        int t[3] = {state[6], state[7], state[8]};
        state[6]  = state[17]; state[7]  = state[14]; state[8]  = state[11];
        state[17] = state[38]; state[14] = state[37]; state[11] = state[36];
        state[38] = state[27]; state[37] = state[30]; state[36] = state[33];
        state[27] = t[0];      state[30] = t[1];      state[33] = t[2];
    }

    void move_L() {
        rotate_face_cw(1);
        int t[3] = {state[0], state[3], state[6]};
        state[0]  = state[53]; state[3]  = state[50]; state[6]  = state[47];
        state[53] = state[36]; state[50] = state[39]; state[47] = state[42];
        state[36] = state[18]; state[39] = state[21]; state[42] = state[24];
        state[18] = t[0];      state[21] = t[1];      state[24] = t[2];
    }

    void move_R() {
        rotate_face_cw(3);
        int t[3] = {state[2], state[5], state[8]};
        state[2]  = state[20]; state[5]  = state[23]; state[8]  = state[26];
        state[20] = state[38]; state[23] = state[41]; state[26] = state[44];
        state[38] = state[51]; state[41] = state[48]; state[44] = state[45];
        state[51] = t[0];      state[48] = t[1];      state[45] = t[2];
    }

    void move_B() {
        rotate_face_cw(5);
        int t[3] = {state[0], state[1], state[2]};
        state[0]  = state[29]; state[1]  = state[32]; state[2]  = state[35];
        state[29] = state[42]; state[32] = state[43]; state[35] = state[44];
        state[42] = state[15]; state[43] = state[12]; state[44] = state[9];
        state[15] = t[0];      state[12] = t[1];      state[9]  = t[2];
    }

    // Derived Moves (Prime and Double)
    void move_Up() { move_U(); move_U(); move_U(); }
    void move_U2() { move_U(); move_U(); }
    void move_Lp() { move_L(); move_L(); move_L(); }
    void move_L2() { move_L(); move_L(); }
    void move_Fp() { move_F(); move_F(); move_F(); }
    void move_F2() { move_F(); move_F(); }
    void move_Rp() { move_R(); move_R(); move_R(); }
    void move_R2() { move_R(); move_R(); }
    void move_Dp() { move_D(); move_D(); move_D(); }
    void move_D2() { move_D(); move_D(); }
    void move_Bp() { move_B(); move_B(); move_B(); }
    void move_B2() { move_B(); move_B(); }

    void execute_move_id(int m) {
        switch(m) {
            case 0: move_U(); break; case 1: move_Up(); break; case 2: move_U2(); break;
            case 3: move_L(); break; case 4: move_Lp(); break; case 5: move_L2(); break;
            case 6: move_F(); break; case 7: move_Fp(); break; case 8: move_F2(); break;
            case 9: move_R(); break; case 10: move_Rp(); break; case 11: move_R2(); break;
            case 12: move_D(); break; case 13: move_Dp(); break; case 14: move_D2(); break;
            case 15: move_B(); break; case 16: move_Bp(); break; case 17: move_B2(); break;
        }
    }

    int* get_state() { return state; }
}
