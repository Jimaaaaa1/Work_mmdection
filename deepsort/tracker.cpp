#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif
#include "tracker.hpp"
#include <queue>
unsigned int now()
{
#ifdef _WIN32
    return GetTickCount();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}

int Tracker::global_id = 0;

Tracker::Tracker(const Ort::Box &b, int n, float noise) : KF(n, 2, 0), s{b}
{
    i = global_id++;
    /**
     * CV_PROP_RW Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
     * CV_PROP_RW Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
     * CV_PROP_RW Mat transitionMatrix;   //!< state transition matrix (A)
     * CV_PROP_RW Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
     * CV_PROP_RW Mat measurementMatrix;  //!< measurement matrix (H)
     * CV_PROP_RW Mat processNoiseCov;    //!< process noise covariance matrix (Q)
     * CV_PROP_RW Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
     * CV_PROP_RW Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)
     * CV_PROP_RW Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
     * CV_PROP_RW Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
     */
    s = b;
    // KF.statePre.setTo(0);
    // KF.statePost.setTo(0);
    // KF.statePre.at<float>(0) = (b.x1 + b.x2) / 2;
    // KF.statePre.at<float>(1) = (b.y1 + b.y2) / 2;
    // KF.statePost.at<float>(0) = (b.x1 + b.x2) / 2;
    // KF.statePost.at<float>(1) = (b.y1 + b.y2) / 2;

    cv::setIdentity(KF.measurementMatrix);
    cv::Mat tmp = KF.measurementMatrix.clone();
    cv::randn(tmp, cv::Scalar::all(0), cv::Scalar::all(1e-5));
    KF.measurementMatrix += tmp;

    cv::setIdentity(KF.transitionMatrix);
    tmp = KF.transitionMatrix.clone();
    cv::randn(tmp, cv::Scalar::all(0), cv::Scalar::all(1e-5));
    KF.transitionMatrix += tmp;

    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(noise));

    predict();
    update(b);
    predict();
    update(b);
}

void Tracker::predict()
{
    lost += 1;
    KF.predict();
    cv::Mat preret = KF.measurementMatrix * KF.statePre;

    auto w = s.x2 - s.x1;
    auto h = s.y2 - s.y1;
    s.x1 = preret.at<float>(0) - w / 2;
    s.x2 = preret.at<float>(0) + w / 2;
    s.y1 = preret.at<float>(1) - h / 2;
    s.y2 = preret.at<float>(1) + h / 2;
}

void Tracker::update(Ort::Box box, bool reset_loss)
{
    if (reset_loss)
        lost = 0;
    if (s.cls == -1)
        s.cls = box.cls; // first time
    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
    measurement.at<float>(0) = (box.x1 + box.x2) / 2;
    measurement.at<float>(1) = (box.y1 + box.y2) / 2;
    KF.correct(measurement);
    cv::Mat ret = KF.measurementMatrix * KF.statePost;

    auto w = box.x2 - box.x1;
    auto h = box.y2 - box.y1;
    s.x1 = ret.at<float>(0) - w / 2;
    s.x2 = ret.at<float>(0) + w / 2;
    s.y1 = ret.at<float>(1) - h / 2;
    s.y2 = ret.at<float>(1) + h / 2;
}

float Tracker::cost(Ort::Box box)
{
    return 1 - s.iou(box);
    // a simple cost function
    auto dx = (box.x1 + box.x2) / 2 - (s.x1 + s.x2) / 2;
    auto dy = (box.y1 + box.y2) / 2 - (s.y1 + s.y2) / 2;
    auto dw = (box.x2 - box.x1) - (s.x2 - s.x1);
    auto dh = (box.y2 - box.y1) - (s.y2 - s.y1);
    return std::sqrt(dx * dx + dy * dy) + std::sqrt(dw * dw + dh * dh);
}

bool AllTrackers::update(const std::vector<Ort::Box> &boxes, float lost)
{
    bool islost = false;
    if (trackers.empty())
    {
        for (auto &box : boxes)
        {
            std::cout << box << std::endl;
            trackers.push_back(Tracker(box, states));
            trackers.back().predict();
            trackers.back().update(box);
            trackers.back().predict();
            trackers.back().update(box);
        }
        return true;
    }
    for (auto &tracker : trackers)
    {
        tracker.predict();
    }

    {
        std::vector<bool> assignedBoxes(boxes.size(), false);
        std::vector<std::vector<double>> cost;
        for (auto &tracker : trackers)
        {
            std::vector<double> tmp;
            for (auto &box : boxes)
            {
                tmp.push_back(tracker.cost(box));
            }
            cost.push_back(tmp);
        }
        std::vector<int> ass;
        hungarian.Solve(cost, ass);

        for (int r = 0; r < ass.size(); ++r)
        {
            if (ass[r] == -1)
            {
                // 没有预测框匹配成功
                trackers[r].update(trackers[r].s, false);
                continue;
            }
            if (trackers[r].cost(boxes[ass[r]]) < 1 - 1e-5)
            {
                // iou > 1e-5
                trackers[r].update(boxes[ass[r]]);
                assignedBoxes[ass[r]] = true;
            }
            else
            {
                // 匹配上，但是iou特别低，视为没匹配上
                trackers[r].update(trackers[r].s, false);
            }
        }

        for (int i = 0; i < assignedBoxes.size(); ++i)
        {
            if (!assignedBoxes[i])
            {
                trackers.push_back(Tracker(boxes[i]));
            }
        }
    }

    trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
                                  [](const Tracker &t)
                                  { return t.lost >= 10; }),
                   trackers.end());
    return islost;
}

void AllTrackers::render(cv::Mat &image)
{
    for (int i = 0; i < trackers.size(); ++i)
    {
        if (trackers[i].s.cls == -1)
            continue;
        auto rect = trackers[i].s.rect();
        cv::rectangle(image, rect, {0, 255, 0}, 2);
        cv::putText(image, std::to_string(trackers[i].i), rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);
    }
    cv::putText(image, fps.fps(now()), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);
}

const char *FPS::fps(decltype(now()) t)
{
    times.push(t);
    if (times.size() < 2)
        return "";
    if (times.size() > 20)
        times.pop();
    sprintf(ss, "%8.2f FPS", times.size() * 1000.0 / (times.back() - times.front()));
    return ss;
}