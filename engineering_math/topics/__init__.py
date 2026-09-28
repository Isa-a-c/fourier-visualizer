"""주제 모듈 등록소. 새 모듈은 이곳에 등록한다."""
def get_topics():
    from .fourier import FourierTopic
    from .heat import HeatTopic
    from .wave import WaveTopic
    return {topic.id:topic for topic in (FourierTopic(),HeatTopic(),WaveTopic())}


def get_topic(topic_id):
    try: return get_topics()[topic_id]
    except KeyError: raise ValueError('등록되지 않은 주제입니다.') from None
