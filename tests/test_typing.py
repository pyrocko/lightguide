from lightguide.blast import Blast

blast = Blast.from_pyrocko([], channel_spacing=1.0)
blast.detrend(type="constant")
