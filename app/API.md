## 统计
### 企业统计
- path: /scholar/standard/statistics/column
- 参数
    - col:必需，统计字段，可以是
      - 地区字段类型：country代表国家,area代表省份地区
      - 分类：classification_mark，标准分类标志：0其他，1国家标准，2地方标准，3行业标准，4团体标准，5企业标准标准，6国外标准，7国际标准，8计量规程标准
      - ccs或者ics：标准分类号，ccs代表国家标准分类号，ics代表国外国家标准分类号
      - publish_org：发布机构
      - category_code：标准分类代号
    - industryId:产业id，非必需，可选1-14
    - area:非必需，查询相关地区的标准统计信息，当column为country时area=country字段，当column为area时是area=area字段
    - year：非必需，默认2002，获取大于等于year的标准统计数据
    - size：返回数据的大小，非必需，默认是10
- 返回：
    - List<Node>
    - Node:{name:col代表统计列的相关值,value:标准数,}
### 地区统计
- path: /scholar/standard/statistics/year
- 参数
    - area:非必需，查询相关地区的标准年份统计信息
    - year：非必需，默认2002，获取大于等于year的标准年份统计数据
- 返回：
  - List<Node>
  - Node:{name:年份,value:标准数,}



## 路径演化
- path: /model/path/evolution
- 返回：
    - data:{List<pos_label>, List<x_axis>, List<y_axis>}
    - List<pos_label>:{每个坐标对应的技术名称}
    - List<x_axis>: {对应的年份，横坐标}
    - List<y_axis>: {对应专利所占百分比，纵坐标}


## 热点预测
- path: /model/hot/prediction?t_name=??
- 参数
    - t_name: 技术名称
- 返回：
    - List<x_org, y_org, x_predict, y_predict>
    - x_org\y_org:{对应统计的年份\每年相关技术专利所占百分比}
    - x_predict\y_predict: {对应统计的年份\每年相关技术专利所占百分比的预测值}

## 热点挖掘
- path: /model/hot/discovery
- 返回：
    - List<Node>
    - Node:{技术名称,该技术的研究热度值,}

## 专利信息查询
### 专利数查询
- path: /patent-query/query/patent-count
- 返回:
  - 专利数量

### 企业数查询
- path: /patent-query/query/enterprise-count
- 返回:
  - 企业数量


### 每个地区企业分布查询（省份）
- path: /patent-query/query/province-enterprise-dict
- 返回:
  - List<Node>
  - Node:{name:省份,List:<企业名>}

### 每个地区企业分布查询（城市）
- path: /patent-query/query/city-enterprise-dict
- 返回:
  - List<Node>
  - Node:{name:城市,List:<企业名>}