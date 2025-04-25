#pragma once

#include <string>
#include <tuple>
#include <vector>
#include <bitset>
#include <array>
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>

#include <GcCore/libCommon/CppNorm.hpp>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

namespace tdns
{
namespace graphics
{
    /*!
    * \brief Structure of a transfer function color point.
    */
    struct controlPoint {
        float value; /*! 0.0 - 1.0 represents the value of physical data at this point */
        float alpha; /*! 0.0 - 1.0 represents the alpha value of the color at this point */
        int index;  /*! index of the control point in the vector */
        glm::vec3 rgb; /*! RGB color of the control point */
    };

    class TDNS_API TransferFunction
    {
    public:
        static constexpr float PointsRadius = 8.f;
        static constexpr float PointSelectionRadius = 12.f;
        static constexpr float SqrPointSelectionRadius =
            PointSelectionRadius * PointSelectionRadius;

    public:
        TransferFunction(int samplesCount); 

        void regenerateSamples();
        int draw(const ImVec2& size = ImVec2(0, 200));
        inline glm::vec4* get_samples_data()
        {
            return samples.data();
        }

    protected:
        [[nodiscard]] glm::vec4 getSample(float xValue) const;
        float lerp(float a, float b, float t) const;
        ImVec2 lerp(const ImVec2& a, const ImVec2& b, const ImVec2& t);
        float invLerp(float a, float b, float v) const;
        ImVec2 invLerp(const ImVec2& a, const ImVec2& b, const ImVec2& v);
        void clamp01(ImVec2& position);
        float sqrDistance(const ImVec2& a, const ImVec2& b);
        ImVec2 getPointPosition(const controlPoint& values, const ImRect& bb);
        std::tuple<float, float> getPointValues(const ImVec2& position, const ImRect& bb);
        bool isValidPoint(const controlPoint point);



    private:
        std::vector<controlPoint> curvesControlPoints; /*!< List of control points */
        std::vector<std::tuple<int, float, float>>  histogram; /*!< Histogram of the data */
        controlPoint hoveredPoint; /*!< Currently hovered control point */
        controlPoint draggedPoint; /*!< Currently dragged control point */
        controlPoint editedPoint; /*!< Currently edited control point */

        void drawHeader();
        void drawCurve(ImDrawList* drawList, const ImRect& bb);

        void handleHoveredPoint(const ImRect& bb);
        bool handlerInputs(const ImRect& bb);

        std::vector<glm::vec4> samples;
        int samplesCount;
        int dataCount;
    };
} // namespace graphics
} // namespace tdns
