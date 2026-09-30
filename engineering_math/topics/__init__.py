"""주제 모듈 등록소. 새 모듈은 이곳에 등록한다."""
def get_topics():
    from .fourier import FourierTopic
    from .heat import HeatTopic
    from .wave import WaveTopic
    from .ode import OdeTopic
    from .laplace import LaplaceTopic
    from .chapter11_series import HalfRangeTopic, ForcedTopic, ApproximationTopic, SturmTopic, OrthogonalTopic
    from .chapter11_transforms import IntegralTopic, SineCosineTopic, ContinuousTransformTopic, TransformTableTopic
    from .chapter12 import (PdeBasicsTopic, DalembertTopic, InfiniteHeatTopic, MembraneTopic,
                            DiskMembraneTopic, PotentialTopic, PdeLaplaceTopic)
    ordered = (FourierTopic(), HalfRangeTopic(), ForcedTopic(), ApproximationTopic(),
               SturmTopic(), OrthogonalTopic(), IntegralTopic(), SineCosineTopic(),
               ContinuousTransformTopic(), TransformTableTopic(),
               PdeBasicsTopic(), WaveTopic(), DalembertTopic(), HeatTopic(), InfiniteHeatTopic(),
               MembraneTopic(), DiskMembraneTopic(), PotentialTopic(), PdeLaplaceTopic(),
               OdeTopic(), LaplaceTopic())
    return {topic.id: topic for topic in ordered}


def get_topic(topic_id):
    try: return get_topics()[topic_id]
    except KeyError: raise ValueError('등록되지 않은 주제입니다.') from None
