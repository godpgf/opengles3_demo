#include <stdlib.h>
#include "esUtil.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


typedef struct {
    // Handle to a program object
    GLuint programObject;

    // Guide Texture
    GLint matChangeColor;
    GLint rSlopes, gSlopes, bSlopes;
    GLint rShifts, gShifts, bShifts;

    // Source and Coeffs Texture
    GLint cwId, cbId;
    GLuint textureId, textrueIdL;
    GLuint coeff_rId, coeff_rIdL;
    GLuint coeff_gId, coeff_gIdL;
    GLuint coeff_bId, coeff_bIdL;

} UserData;


GLubyte *LoadPic(const basic_string<char> &filename, int &width, int &height) {
    //OpenCV读取图像
    Mat S = imread(filename);
    Mat I;
    cvtColor(S, I, COLOR_BGR2RGB);

    //设置长宽
    width = I.cols;
    height = I.rows;

    //获取图像指针
    int pixellength = width * height * 3;

    // 2x2 Image, 3 bytes per pixel (R, G, B)
    GLubyte *pixels = new GLubyte[pixellength];
    memcpy(pixels, I.data, pixellength * sizeof(char));
    return pixels;
}

GLuint CreateSourceTexture2D(GLubyte *pixels, int width, int height){
    // Texture object handle
    GLuint textureId;
    // Use tightly packed data
    glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

    // Generate a texture object
    glGenTextures ( 1, &textureId );

    // Bind the texture object
    glBindTexture ( GL_TEXTURE_2D, textureId );

    // Load the texture
    glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels );

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    return textureId;
}


GLuint CreateCoeffsTexture3D(GLubyte *pixels, int width=16, int height=16, int depth=8){
    // Texture object handle
    GLuint textureId;
    // Use tightly packed data
    glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

    // Generate a texture object
    glGenTextures ( 1, &textureId );

    // Bind the texture object
    glBindTexture ( GL_TEXTURE_3D, textureId );

    // Load the texture
    glTexImage3D ( GL_TEXTURE_3D, 0, GL_RGBA, width, height, depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels );

    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri ( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri ( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    return textureId;
}


///
// Initialize the shader and program object
//
int Init(ESContext *esContext) {
    UserData *userData = (UserData *) esContext->userData;
    char vShaderStr[] =
            "#version 300 es                            \n"
            "layout(location = 0) in vec4 a_position;   \n"
            "layout(location = 1) in vec2 a_texCoord;   \n"
            "out vec2 v_texCoord;                       \n"
            "void main()                                \n"
            "{                                          \n"
            "   gl_Position = a_position;               \n"
            "   v_texCoord = a_texCoord;                \n"
            "}                                          \n";

    char fShaderStr[] =
            "#version 300 es                                                                \n"
            "precision mediump float;                                                       \n"
            "in vec2 v_texCoord;                                                            \n"
            "layout(location = 0) out vec4 outColor;                                        \n"
            "uniform sampler2D s_texture;                                                   \n"
            "uniform mat4 mat_ChangeColor;                                                  \n"
            "uniform mat4 r_slopes;                                                         \n"
            "uniform mat4 g_slopes;                                                         \n"
            "uniform mat4 b_slopes;                                                         \n"
            "uniform mat4 r_shifts;                                                         \n"
            "uniform mat4 g_shifts;                                                         \n"
            "uniform mat4 b_shifts;                                                         \n"
            "uniform mediump sampler3D coeff_r;                                             \n"
            "uniform mediump sampler3D coeff_g;                                             \n"
            "uniform mediump sampler3D coeff_b;                                             \n"
            "uniform float coeff_w;                                                         \n"
            "uniform float coeff_b;                                                         \n"
            "const vec4 v1 = vec4(1.0);                                                     \n"
            "mat4 mabs(mat4 a){                                                             \n"
            "  mat4 b = mat4(abs(a[0][0]), abs(a[0][1]), abs(a[0][2]), abs(a[0][3]),        \n"
            "                abs(a[1][0]), abs(a[1][1]), abs(a[1][2]), abs(a[1][3]),        \n"
            "                abs(a[2][0]), abs(a[2][1]), abs(a[2][2]), abs(a[2][3]),        \n"
            "                abs(a[3][0]), abs(a[3][1]), abs(a[3][2]), abs(a[3][3]));       \n"
            "  return b;                                                                    \n"
            "}                                                                              \n"
            "void main(){                                                                   \n"
            "  vec4 nColor = texture( s_texture, v_texCoord );                              \n"
            "  vec4 gColor = nColor * mat_ChangeColor;                                      \n"
            "  vec4 vr = v1 * matrixCompMult(r_slopes, mabs(gColor.r - r_shifts));          \n"
            "  float r = vr.r + vr.g + vr.b + vr.a;                                         \n"
            "  vec4 vg = v1 * matrixCompMult(g_slopes, mabs(gColor.g - g_shifts));          \n"
            "  float g = vg.r + vg.g + vg.b + vg.a;                                         \n"
            "  vec4 vb = v1 * matrixCompMult(b_slopes, mabs(gColor.b - b_shifts));          \n"
            "  float b = vb.r + vb.g + vb.b + vb.a;                                         \n"
            "  float guide = r + g + b;                                                     \n"
            "  vec3 v_texCoord3d = vec3(v_texCoord.xy, guide);                              \n"
            "  vec4 cr = (texture(coeff_r, v_texCoord3d) * coeff_w + coeff_b) * nColor;     \n"
            "  vec4 cg = (texture(coeff_g, v_texCoord3d) * coeff_w + coeff_b) * nColor;     \n"
            "  vec4 cb = (texture(coeff_b, v_texCoord3d) * coeff_w + coeff_b) * nColor;     \n"
            "  outColor = vec4(dot(cr, dc), dot(cg, dc), dot(cb, dc), 1.0);                 \n"
            "}                                                                              \n";

    // Load the shaders and get a linked program object
    userData->programObject = esLoadProgram(vShaderStr, fShaderStr);

    userData->matChangeColor = glGetUniformLocation(userData->programObject, "mat_ChangeColor");
    userData->rShifts = glGetUniformLocation(userData->programObject, "r_shifts");
    userData->gShifts = glGetUniformLocation(userData->programObject, "g_shifts");
    userData->bShifts = glGetUniformLocation(userData->programObject, "b_shifts");
    userData->rSlopes = glGetUniformLocation(userData->programObject, "r_slopes");
    userData->gSlopes = glGetUniformLocation(userData->programObject, "g_slopes");
    userData->bSlopes = glGetUniformLocation(userData->programObject, "b_slopes");

    // Load the texture
    int width, height;
    GLubyte *pixels = LoadPic("D:\\le_pic\\dataset\\dataset_5k_1\\output\\a0417-IMG_3345.jpg", width, height);
    userData->textureId = CreateSourceTexture2D(pixels, width, height);
    userData->textrueIdL = glGetUniformLocation(userData->programObject, "s_texture");

    userData->coeff_rId = CreateCoeffsTexture3D(nullptr);
    userData->coeff_rIdL = glGetUniformLocation(userData->programObject, "coeff_r");
    userData->coeff_gId = CreateCoeffsTexture3D(nullptr);
    userData->coeff_gIdL = glGetUniformLocation(userData->programObject, "coeff_g");
    userData->coeff_bId = CreateCoeffsTexture3D(nullptr);
    userData->coeff_bIdL = glGetUniformLocation(userData->programObject, "coeff_b");
    userData->cwId = glGetUniformLocation(userData->programObject, "coeff_w");
    userData->cbId = glGetUniformLocation(userData->programObject, "coeff_b");

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    return TRUE;
}

///
// Draw a triangle using the shader pair created in Init()
//
void Draw(ESContext *esContext) {
    UserData *userData = (UserData *) esContext->userData;
    GLfloat vVertices[] = {-1.0f, 1.0f, 0.0f,  // Position 0
                           0.0f, 0.0f,        // TexCoord 0
                           -1.0f, -1.0f, 0.0f,  // Position 1
                           0.0f, 1.0f,        // TexCoord 1
                           1.0f, -1.0f, 0.0f,  // Position 2
                           1.0f, 1.0f,        // TexCoord 2
                           1.0f, 1.0f, 0.0f,  // Position 3
                           1.0f, 0.0f         // TexCoord 3
    };
    GLushort indices[] = {0, 1, 2, 0, 2, 3};

    // Set the viewport
    glViewport(0, 0, esContext->width, esContext->height);

    // Clear the color buffer
    glClear(GL_COLOR_BUFFER_BIT);

    // Use the program object
    glUseProgram(userData->programObject);

    // Load the vertex position
    glVertexAttribPointer(0, 3, GL_FLOAT,
                          GL_FALSE, 5 * sizeof(GLfloat), vVertices);
    // Load the texture coordinate
    glVertexAttribPointer(1, 2, GL_FLOAT,
                          GL_FALSE, 5 * sizeof(GLfloat), &vVertices[3]);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    // Bind the texture -----------------------------------------------------------------

    // Set the sampler texture unit to 0
    glUniform1i(userData->textrueIdL, GL_TEXTURE0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, userData->textureId);

    glUniform1i(userData->coeff_rIdL, GL_TEXTURE1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, userData->coeff_rId);

    glUniform1i(userData->coeff_gIdL, GL_TEXTURE2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, userData->coeff_gId);

    glUniform1i(userData->coeff_bIdL, GL_TEXTURE3);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, userData->coeff_bId);

    // set color
    float coffs[16 * 7] = {
            //matChangeColor
            1.017366f, 0.0317081f, 0.02311327f, 0.03549889f,
            0.14991048f, 1.0467575f, 0.12823085f, 0.0308079f,
            0.07468443f, 0.05145404f, 0.9556362f, 0.03285963f,
            0.f, 0.f, 0.f, 1.f,
            //rSlopes
            1.0085183f, -0.05950956f, -0.0594769f, -0.02936393f,
            -0.02918992f, -0.02299553f, -0.01945908f, -0.01277439f,
            -0.00916772f, -0.00398887f, -0.00881052f, -0.02558564f,
            0.02407969f, 0.04131978f, 0.04463151f, -0.0972411f,
            //gSlopes
            1.0072497f, -0.06259239f, -0.06298216f, -0.04867213f,
            -0.0537422f, -0.05448569f, -0.05622008f, -0.058614f,
            -0.05571636f, -0.0521814f, -0.03497815f, -0.0530565f,
            -0.0657321f, -0.07108345f, -0.07943164f, -0.21914218f,
            //bSlopes
            0.98447245f, -0.08041759f, -0.08125997f, -0.0787712f,
            -0.07707487f, -0.07032022f, -0.06528138f, -0.05689945f,
            -0.03600536f, 0.07634532f, 0.06367546f, 0.05874879f,
            0.03069806f, -0.00104325f, -0.03076693f, -0.13997602f,
            //rShifts
            -0.03144173f, 0.06935108f, 0.06935094f, 0.16436048f,
            0.1746024f, 0.18639989f, 0.19805253f, 0.24173449f,
            0.3180594f, 0.35885164f, 0.5388886f, 0.55644774f,
            0.7281175f, 0.7296287f, 0.82252324f, 0.7157709f,
            //gShifts
            -0.03140695f, 0.07639092f, 0.07638453f, 0.1922387f,
            0.1974331f, 0.20506965f, 0.20823415f, 0.21551362f,
            0.2547321f, 0.33319432f, 0.4282976f, 0.60960686f,
            0.61081827f, 0.63159513f, 0.7371784f, 0.6891297f,
            //bShifts
            -0.03169918f, 0.06509382f, 0.06509269f, 0.14744808f,
            0.14919801f, 0.15371335f, 0.159195f, 0.16920638f,
            0.1880372f, 0.4327274f, 0.5185255f, 0.62188935f,
            0.7129958f, 0.81431925f, 0.85236645f, 0.8537769f
    };
    float *p = coffs;
    glUniformMatrix4fv(userData->matChangeColor, 1, GL_FALSE, p);
    p += 16;
    glUniformMatrix4fv(userData->rSlopes, 1, GL_FALSE, p);
    p += 16;
    glUniformMatrix4fv(userData->gSlopes, 1, GL_FALSE, p);
    p += 16;
    glUniformMatrix4fv(userData->bSlopes, 1, GL_FALSE, p);
    p += 16;
    glUniformMatrix4fv(userData->rShifts, 1, GL_FALSE, p);
    p += 16;
    glUniformMatrix4fv(userData->gShifts, 1, GL_FALSE, p);
    p += 16;
    glUniformMatrix4fv(userData->bShifts, 1, GL_FALSE, p);

    glUniform1f(userData->cwId, 1.0);
    glUniform1f(userData->cbId, 0.0);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);
}

///
// Cleanup
//
void ShutDown(ESContext *esContext) {
    UserData *userData = (UserData *) esContext->userData;

    // Delete texture object
    glDeleteTextures(1, userData->textureIds);

    // Delete program object
    glDeleteProgram(userData->programObject);
}


int esMain(ESContext *esContext) {
    esContext->userData = malloc(sizeof(UserData));

    esCreateWindow(esContext, "Simple Texture 2D", 320, 240, ES_WINDOW_RGB);

    if (!Init(esContext)) {
        return GL_FALSE;
    }

    esRegisterDrawFunc(esContext, Draw);
    esRegisterShutdownFunc(esContext, ShutDown);

    return GL_TRUE;
}
